"""Haze-aware restoration backbone."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from utils.frequency import high_frequency
from utils.physics import dark_channel, transmission_proxy


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ChannelSpatialGate(nn.Module):
    """Lightweight channel-spatial attention for haze-sensitive features."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_gate = self.channel(x)
        mean = x.mean(dim=1, keepdim=True)
        maximum = x.amax(dim=1, keepdim=True)
        spatial_gate = self.spatial(torch.cat([mean, maximum], dim=1))
        return channel_gate * spatial_gate


class HazeAwareBlock(nn.Module):
    """Residual block with dilated depthwise mixing and optional prior modulation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prior_channels: int = 0,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.depthwise = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            padding=dilation,
            dilation=dilation,
            groups=out_channels,
        )
        self.pointwise = nn.Conv2d(out_channels, out_channels, 1)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.prior = nn.Conv2d(prior_channels, out_channels, 1) if prior_channels else None
        self.gate = ChannelSpatialGate(out_channels)
        self.ffn = nn.Sequential(
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.Conv2d(out_channels, out_channels * 2, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels * 2, out_channels, 1),
        )

    def forward(self, x: torch.Tensor, prior: torch.Tensor | None = None) -> torch.Tensor:
        h = self.conv1(x)
        h = F.silu(self.norm1(h), inplace=True)
        h = self.pointwise(self.depthwise(h))
        h = self.norm2(h)
        if self.prior is not None and prior is not None:
            resized_prior = F.interpolate(prior, size=h.shape[-2:], mode="bilinear", align_corners=False)
            h = h + self.prior(resized_prior)
        h = F.silu(h, inplace=True)
        h = self.skip(x) + h * self.gate(h)
        return h + self.ffn(h)


class GlobalContextBlock(nn.Module):
    """Cheap global context compensation for large non-uniform haze regions."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.proj(x) * self.context(x)


class HazeAwareRestorationUNet(nn.Module):
    """Physics/frequency-modulated U-Net for coarse dehazing.

    The module keeps the U-Net interface but uses dark-channel, transmission,
    and high-frequency hints inside each scale instead of leaving them only to
    the diffusion branch.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 32,
        use_physics: bool = True,
        use_frequency: bool = True,
    ) -> None:
        super().__init__()
        c = base_channels
        self.use_physics = use_physics
        self.use_frequency = use_frequency
        prior_channels = 0
        if use_physics:
            prior_channels += 2
        if use_frequency:
            prior_channels += 3
        self.prior_channels = prior_channels

        self.stem = nn.Conv2d(in_channels + prior_channels, c, 3, padding=1)
        self.enc1 = HazeAwareBlock(c, c, prior_channels=prior_channels, dilation=1)
        self.down1 = nn.Conv2d(c, c * 2, 3, stride=2, padding=1)
        self.enc2 = HazeAwareBlock(c * 2, c * 2, prior_channels=prior_channels, dilation=2)
        self.down2 = nn.Conv2d(c * 2, c * 4, 3, stride=2, padding=1)
        self.enc3 = HazeAwareBlock(c * 4, c * 4, prior_channels=prior_channels, dilation=2)
        self.down3 = nn.Conv2d(c * 4, c * 4, 3, stride=2, padding=1)
        self.mid1 = HazeAwareBlock(c * 4, c * 4, prior_channels=prior_channels, dilation=4)
        self.context = GlobalContextBlock(c * 4)
        self.mid2 = HazeAwareBlock(c * 4, c * 4, prior_channels=prior_channels, dilation=2)

        self.up3 = nn.Conv2d(c * 4, c * 4, 3, padding=1)
        self.dec3 = HazeAwareBlock(c * 8, c * 2, prior_channels=prior_channels, dilation=2)
        self.up2 = nn.Conv2d(c * 2, c * 2, 3, padding=1)
        self.dec2 = HazeAwareBlock(c * 4, c, prior_channels=prior_channels, dilation=1)
        self.up1 = nn.Conv2d(c, c, 3, padding=1)
        self.dec1 = HazeAwareBlock(c * 2, c, prior_channels=prior_channels, dilation=1)
        self.out = nn.Conv2d(c, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prior = self._build_prior(x)
        stem_input = torch.cat([x, prior], dim=1) if prior is not None else x
        e1 = self.enc1(self.stem(stem_input), prior)
        e2 = self.enc2(self.down1(e1), prior)
        e3 = self.enc3(self.down2(e2), prior)
        mid = self.mid1(self.down3(e3), prior)
        mid = self.mid2(self.context(mid), prior)
        d3 = F.interpolate(mid, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([self.up3(d3), e3], dim=1), prior)
        d2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([self.up2(d2), e2], dim=1), prior)
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([self.up1(d1), e1], dim=1), prior)
        residual = 0.5 * torch.tanh(self.out(d1))
        return (x + residual).clamp(0.0, 1.0)

    def _build_prior(self, x: torch.Tensor) -> torch.Tensor | None:
        parts = []
        if self.use_physics:
            parts.extend([dark_channel(x), transmission_proxy(x)])
        if self.use_frequency:
            parts.append(high_frequency(x))
        if not parts:
            return None
        return torch.cat(parts, dim=1)


def _groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1
