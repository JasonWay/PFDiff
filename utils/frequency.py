"""Frequency and high-pass helpers."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def high_frequency(image: torch.Tensor) -> torch.Tensor:
    """Return a Laplacian high-frequency residual."""

    channels = image.shape[1]
    kernel = image.new_tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]])
    kernel = kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    return F.conv2d(image, kernel, padding=1, groups=channels)
