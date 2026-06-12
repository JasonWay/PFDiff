"""Torch image conversion helpers."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().float().cpu().clamp(0.0, 1.0)
    if tensor.ndim == 4:
        tensor = tensor[0]
    tensor = (tensor * 255.0).round().byte()
    tensor = tensor.permute(1, 2, 0).contiguous()
    height, width = tensor.shape[:2]
    return Image.frombytes("RGB", (width, height), tensor.numpy().tobytes())


def save_tensor_image(tensor: torch.Tensor, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tensor_to_pil(tensor).save(output)
