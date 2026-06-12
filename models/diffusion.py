"""Conditional residual diffusion modules."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .backbone import ConvBlock


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(inplace=True),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = t.device
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, device=device).float() / max(half - 1, 1)
        )
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))
        return self.mlp(emb)


class ConditionPyramid(nn.Module):
    """Encode restoration/physics/frequency conditions at denoiser scales."""

    def __init__(self, condition_channels: int, base_channels: int) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock(condition_channels, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.mid = ConvBlock(c * 4, c * 4)

    def forward(self, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c1 = self.enc1(condition)
        c2 = self.enc2(F.avg_pool2d(c1, 2))
        c3 = self.enc3(F.avg_pool2d(c2, 2))
        cm = self.mid(F.avg_pool2d(c3, 2))
        return c1, c2, c3, cm


class ModulatedTimeBlock(nn.Module):
    """Residual denoising block with timestep and condition FiLM modulation."""

    def __init__(self, in_channels: int, out_channels: int, cond_channels: int, time_dim: int) -> None:
        super().__init__()
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.time_proj = nn.Linear(time_dim, out_channels * 2)
        self.cond_proj = nn.Conv2d(cond_channels, out_channels * 2, 1)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, max(out_channels // 4, 8), 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(out_channels // 4, 8), out_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm1(self.conv1(x))
        if cond.shape[-2:] != h.shape[-2:]:
            cond = F.interpolate(cond, size=h.shape[-2:], mode="bilinear", align_corners=False)
        scale_shift = self.time_proj(time_emb)[:, :, None, None] + self.cond_proj(cond)
        scale, shift = scale_shift.chunk(2, dim=1)
        h = h * (1.0 + scale) + shift
        h = F.silu(h, inplace=True)
        h = F.silu(self.norm2(self.conv2(h)), inplace=True)
        h = h * self.attn(h)
        return self.skip(x) + h


class ConditionalDenoiseUNet(nn.Module):
    """Denoiser with cross-scale condition injection."""

    def __init__(
        self,
        residual_channels: int = 3,
        condition_channels: int = 6,
        base_channels: int = 32,
        time_dim: int = 128,
    ) -> None:
        super().__init__()
        c = base_channels
        self.time = SinusoidalTimeEmbedding(time_dim)
        self.condition = ConditionPyramid(condition_channels, c)
        self.enc1 = ModulatedTimeBlock(residual_channels, c, c, time_dim)
        self.enc2 = ModulatedTimeBlock(c, c * 2, c * 2, time_dim)
        self.enc3 = ModulatedTimeBlock(c * 2, c * 4, c * 4, time_dim)
        self.mid = ModulatedTimeBlock(c * 4, c * 4, c * 4, time_dim)
        self.dec3 = ModulatedTimeBlock(c * 8, c * 2, c * 4, time_dim)
        self.dec2 = ModulatedTimeBlock(c * 4, c, c * 2, time_dim)
        self.dec1 = ModulatedTimeBlock(c * 2, c, c, time_dim)
        self.out = nn.Conv2d(c, residual_channels, 3, padding=1)

    def forward(self, residual_t: torch.Tensor, condition: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        time_emb = self.time(t)
        c1, c2, c3, cm = self.condition(condition)
        e1 = self.enc1(residual_t, time_emb, c1)
        e2 = self.enc2(F.avg_pool2d(e1, 2), time_emb, c2)
        e3 = self.enc3(F.avg_pool2d(e2, 2), time_emb, c3)
        mid = self.mid(F.avg_pool2d(e3, 2), time_emb, cm)
        d3 = F.interpolate(mid, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1), time_emb, c3)
        d2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1), time_emb, c2)
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1), time_emb, c1)
        return self.out(d1)


class GaussianResidualDiffusion(nn.Module):
    """DDPM training and DDIM-style sampling for residual refinement."""

    def __init__(
        self,
        denoiser: nn.Module,
        timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ) -> None:
        super().__init__()
        self.denoiser = denoiser
        self.timesteps = timesteps
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def training_loss(
        self,
        residual: torch.Tensor,
        condition: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        batch = residual.shape[0]
        t = torch.randint(0, self.timesteps, (batch,), device=residual.device)
        noise = torch.randn_like(residual) if noise is None else noise
        residual_t = self.q_sample(residual, t, noise)
        pred_noise = self.denoiser(residual_t, condition, t)
        noise_loss = F.mse_loss(pred_noise, noise)
        if not return_details:
            return noise_loss
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1)
        pred_residual = (residual_t - (1.0 - alpha_bar).sqrt() * pred_noise) / alpha_bar.sqrt().clamp_min(1e-8)
        return {
            "loss_noise": noise_loss,
            "pred_noise": pred_noise,
            "pred_residual": pred_residual,
            "target_residual": residual,
        }

    def q_sample(self, residual: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1)
        return alpha_bar.sqrt() * residual + (1.0 - alpha_bar).sqrt() * noise

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        shape: torch.Size,
        steps: int = 10,
        start: str = "random",
    ) -> torch.Tensor:
        if start == "zeros":
            residual = torch.zeros(shape, device=condition.device)
        elif start == "random":
            residual = torch.randn(shape, device=condition.device)
        else:
            raise ValueError(f"Unsupported diffusion sample start: {start}")
        step_ids = torch.linspace(self.timesteps - 1, 0, steps, device=condition.device).long()
        for index, t_scalar in enumerate(step_ids):
            t = torch.full((shape[0],), int(t_scalar.item()), device=condition.device, dtype=torch.long)
            eps = self.denoiser(residual, condition, t)
            alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1)
            pred_x0 = (residual - (1.0 - alpha_bar).sqrt() * eps) / alpha_bar.sqrt().clamp_min(1e-8)
            if index == len(step_ids) - 1:
                residual = pred_x0
            else:
                t_next = step_ids[index + 1].long()
                alpha_next = self.alpha_bars[t_next].view(1, 1, 1, 1)
                residual = alpha_next.sqrt() * pred_x0 + (1.0 - alpha_next).sqrt() * eps
        return residual


def _groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1
