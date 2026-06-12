#!/usr/bin/env python
"""Run PFDiff-Dehaze inference on the merged StateHaze1K test split."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

if not os.environ.get("OMP_NUM_THREADS", "1").isdigit():
    os.environ["OMP_NUM_THREADS"] = "1"

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required for testing. Run this on the CUDA/PyTorch remote environment.") from exc

from datasets.statehaze import StateHazeDataset, list_statehaze_pairs
from models import PFDiffDehaze
from utils.config import load_config
from utils.torch_image import save_tensor_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pfdiff_full.json")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None, help='Override compute device, e.g. "cuda:0" or "cpu".')
    parser.add_argument("--sampling-steps", type=int, default=None, help="Override model.sampling_steps for inference.")
    parser.add_argument("--residual-scale", type=float, default=None, help="Override model.residual_scale for inference.")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    requested_config = load_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint.get("config", requested_config)
    _deep_update(config, requested_config)
    if args.sampling_steps is not None:
        config.setdefault("model", {})["sampling_steps"] = args.sampling_steps
    if args.residual_scale is not None:
        config.setdefault("model", {})["residual_scale"] = args.residual_scale
    data_root = Path(args.data_root or config["data"]["root"])
    result_dir = Path(args.output_dir or config["experiment"]["result_dir"])
    output_dir = result_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PFDiffDehaze(
        base_channels=int(config["model"]["base_channels"]),
        diffusion_steps=int(config["model"]["diffusion_steps"]),
        sampling_steps=int(config["model"]["sampling_steps"]),
        residual_scale=float(config["model"].get("residual_scale", 1.0)),
        sampling_start=str(config["model"].get("sampling_start", "random")),
        use_adaptive_residual=bool(config["model"].get("use_adaptive_residual", True)),
        use_diffusion=bool(config["model"]["use_diffusion"]),
        use_physics=bool(config["model"]["use_physics"]),
        use_frequency=bool(config["model"]["use_frequency"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    pairs = list_statehaze_pairs(data_root, "test")
    loader = DataLoader(
        StateHazeDataset(pairs, image_size=None, training=False),
        batch_size=1,
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 2)),
    )
    metric_rows = []
    for batch in loader:
        hazy = batch["hazy"].to(device)
        clear = batch["clear"].to(device)
        pred = model(hazy)
        key = batch["key"][0]
        save_tensor_image(pred[0], output_dir / f"{key}.png")
        mse = torch.mean((pred - clear) ** 2, dim=(1, 2, 3)).clamp_min(1e-12)
        psnr = float((-10.0 * torch.log10(mse))[0].detach().cpu())
        ssim = float(_global_ssim_per_image(pred, clear)[0].detach().cpu())
        metric_rows.append({"key": key, "psnr": psnr, "ssim": ssim})

    metrics_csv = result_dir / "test_metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["key", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(metric_rows)
    avg_psnr = sum(row["psnr"] for row in metric_rows) / len(metric_rows)
    avg_ssim = sum(row["ssim"] for row in metric_rows) / len(metric_rows)
    summary_path = result_dir / "summary.md"
    summary_path.write_text(
        "\n".join(
            [
                "# Test Summary",
                "",
                f"- images: {len(metric_rows)}",
                f"- PSNR: {avg_psnr:.4f}",
                f"- SSIM: {avg_ssim:.6f}",
                f"- metrics_csv: `{metrics_csv}`",
                f"- images_dir: `{output_dir}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"saved_images={len(pairs)}")
    print(f"output_dir={output_dir}")
    print(f"psnr={avg_psnr:.4f}")
    print(f"ssim={avg_ssim:.6f}")
    print(f"metrics_csv={metrics_csv}")
    print(f"summary={summary_path}")


def _global_ssim_per_image(pred: torch.Tensor, clear: torch.Tensor) -> torch.Tensor:
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
    return (numerator / denominator.clamp_min(1e-12)).mean(dim=1)


def _deep_update(target: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


if __name__ == "__main__":
    main()
