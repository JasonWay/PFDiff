"""PFDiff-Dehaze model: physics- and frequency-guided residual diffusion."""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F

from models.backbone import HazeAwareRestorationUNet
from models.diffusion import ConditionalDenoiseUNet, GaussianResidualDiffusion
from utils.frequency import high_frequency
from utils.physics import dark_channel, transmission_proxy


class PFDiffDehaze(nn.Module):
    def __init__(
        self,
        base_channels: int = 32,
        diffusion_steps: int = 100,
        sampling_steps: int = 10,
        residual_scale: float = 1.0,
        sampling_start: str = "random",
        use_adaptive_residual: bool = True,
        use_diffusion: bool = True,
        use_physics: bool = True,
        use_frequency: bool = True,
    ) -> None:
        super().__init__()
        self.use_diffusion = use_diffusion
        self.use_physics = use_physics
        self.use_frequency = use_frequency
        self.use_adaptive_residual = use_adaptive_residual
        self.sampling_steps = sampling_steps
        self.residual_scale = residual_scale
        self.sampling_start = sampling_start
        self.backbone = HazeAwareRestorationUNet(
            base_channels=base_channels,
            use_physics=use_physics,
            use_frequency=use_frequency,
        )

        condition_channels = 6
        if use_physics:
            condition_channels += 2
        if use_frequency:
            condition_channels += 3
        self.condition_channels = condition_channels

        if use_diffusion:
            denoiser = ConditionalDenoiseUNet(
                residual_channels=3,
                condition_channels=condition_channels,
                base_channels=base_channels,
            )
            self.diffusion = GaussianResidualDiffusion(denoiser, timesteps=diffusion_steps)
        else:
            self.diffusion = None
        self.residual_gate = (
            AdaptiveResidualGate(condition_channels, base_channels)
            if use_adaptive_residual and use_diffusion
            else None
        )

    def build_condition(self, hazy: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        parts = [hazy, coarse]
        if self.use_physics:
            parts.extend([dark_channel(hazy), transmission_proxy(hazy)])
        if self.use_frequency:
            parts.append(high_frequency(hazy))
        return torch.cat(parts, dim=1)

    def training_losses(
        self,
        hazy: torch.Tensor,
        clear: torch.Tensor,
        weights: Dict[str, float],
    ) -> Dict[str, torch.Tensor]:
        coarse = self.backbone(hazy)
        l1_loss = F.l1_loss(coarse, clear)
        total = weights.get("l1", 1.0) * l1_loss
        losses: Dict[str, torch.Tensor] = {
            "loss_l1": l1_loss,
        }

        if weights.get("mse", 0.0) > 0:
            mse_loss = F.mse_loss(coarse, clear)
            losses["loss_mse"] = mse_loss
            total = total + weights.get("mse", 0.0) * mse_loss

        if weights.get("ssim", 0.0) > 0:
            ssim_loss = _global_ssim_loss(coarse, clear)
            losses["loss_ssim"] = ssim_loss
            total = total + weights.get("ssim", 0.0) * ssim_loss

        if self.use_frequency and weights.get("frequency", 0.0) > 0:
            freq_loss = F.l1_loss(high_frequency(coarse), high_frequency(clear))
            losses["loss_frequency"] = freq_loss
            total = total + weights.get("frequency", 0.0) * freq_loss

        if self.use_diffusion and self.diffusion is not None:
            condition = self.build_condition(hazy, coarse)
            residual = clear - coarse
            needs_details = (
                self.use_adaptive_residual
                or weights.get("residual_x0", 0.0) > 0
                or weights.get("final", 0.0) > 0
            )
            diffusion_result = self.diffusion.training_loss(residual, condition, return_details=needs_details)
            if isinstance(diffusion_result, dict):
                diffusion_loss = diffusion_result["loss_noise"]
                pred_residual = diffusion_result["pred_residual"]
            else:
                diffusion_loss = diffusion_result
                pred_residual = None
            losses["loss_diffusion"] = diffusion_loss
            total = total + weights.get("diffusion", 0.1) * diffusion_loss
            if pred_residual is not None and weights.get("residual_x0", 0.0) > 0:
                residual_x0_loss = F.l1_loss(pred_residual, residual)
                losses["loss_residual_x0"] = residual_x0_loss
                total = total + weights.get("residual_x0", 0.0) * residual_x0_loss
            if pred_residual is not None and weights.get("final", 0.0) > 0:
                gate = self._residual_gate(condition)
                refined = (coarse + self.residual_scale * gate * pred_residual).clamp(0.0, 1.0)
                final_loss = F.l1_loss(refined, clear)
                losses["loss_final"] = final_loss
                losses["gate_mean"] = gate.detach().mean()
                total = total + weights.get("final", 0.0) * final_loss

        losses["loss_total"] = total
        losses["coarse_mean"] = coarse.detach().mean()
        return losses

    @torch.no_grad()
    def forward(self, hazy: torch.Tensor) -> torch.Tensor:
        coarse = self.backbone(hazy)
        if not self.use_diffusion or self.diffusion is None:
            return coarse.clamp(0.0, 1.0)
        condition = self.build_condition(hazy, coarse)
        residual = self.diffusion.sample(
            condition,
            shape=coarse.shape,
            steps=self.sampling_steps,
            start=self.sampling_start,
        )
        residual = residual.clamp(-1.0, 1.0)
        residual = self._residual_gate(condition) * residual
        return (coarse + self.residual_scale * residual).clamp(0.0, 1.0)

    def _residual_gate(self, condition: torch.Tensor) -> torch.Tensor:
        if self.residual_gate is None:
            return torch.ones(condition.shape[0], 1, *condition.shape[-2:], device=condition.device, dtype=condition.dtype)
        return self.residual_gate(condition)


class AdaptiveResidualGate(nn.Module):
    """Predict where diffusion residuals should modify the coarse restoration."""

    def __init__(self, condition_channels: int, base_channels: int = 32) -> None:
        super().__init__()
        hidden = max(base_channels // 2, 16)
        self.net = nn.Sequential(
            nn.Conv2d(condition_channels, hidden, 3, padding=1),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, 3, padding=1),
        )
        nn.init.constant_(self.net[-1].bias, -2.0)

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(condition))


def _global_ssim_loss(pred: torch.Tensor, clear: torch.Tensor) -> torch.Tensor:
    c1 = 0.01**2
    c2 = 0.03**2
    pred_flat = pred.flatten(start_dim=2)
    clear_flat = clear.flatten(start_dim=2)
    mean_pred = pred_flat.mean(dim=2)
    mean_clear = clear_flat.mean(dim=2)
    var_pred = pred_flat.var(dim=2, unbiased=False)
    var_clear = clear_flat.var(dim=2, unbiased=False)
    cov = ((pred_flat - mean_pred.unsqueeze(-1)) * (clear_flat - mean_clear.unsqueeze(-1))).mean(dim=2)
    numerator = (2 * mean_pred * mean_clear + c1) * (2 * cov + c2)
    denominator = (mean_pred.square() + mean_clear.square() + c1) * (var_pred + var_clear + c2)
    ssim = (numerator / denominator.clamp_min(1e-12)).mean(dim=1)
    return 1.0 - ssim.mean()


def _groups(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1
