"""Atmospheric-scattering-inspired physics hints."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def dark_channel(image: torch.Tensor, patch_size: int = 15) -> torch.Tensor:
    """Approximate dark channel prior map in [0, 1]."""

    min_rgb = image.min(dim=1, keepdim=True).values
    pad = patch_size // 2
    dark = -F.max_pool2d(-min_rgb, kernel_size=patch_size, stride=1, padding=pad)
    return dark.clamp(0.0, 1.0)


def transmission_proxy(image: torch.Tensor, omega: float = 0.95, patch_size: int = 15) -> torch.Tensor:
    """DCP-style transmission proxy used as soft condition only."""

    dark = dark_channel(image, patch_size=patch_size)
    transmission = 1.0 - omega * dark
    return transmission.clamp(0.05, 1.0)
