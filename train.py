#!/usr/bin/env python
"""Train PFDiff-Dehaze on paired StateHaze1K data."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import random
import sys
from typing import Dict, Optional

if not os.environ.get("OMP_NUM_THREADS", "1").isdigit():
    os.environ["OMP_NUM_THREADS"] = "1"

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError as exc:  # pragma: no cover - local lightweight env.
    raise SystemExit("PyTorch is required for training. Run this on the CUDA/PyTorch remote environment.") from exc

from datasets.statehaze import (
    StateHazeDataset,
    list_statehaze_pairs,
)
from models import PFDiffDehaze
from utils.config import load_config, save_config
from utils.torch_image import save_tensor_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pfdiff_full.json")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--variant", default=None, choices=("baseline", "diffusion", "physics", "frequency", "full"))
    parser.add_argument("--resume", default=None, help='Checkpoint path to resume, or "latest".')
    parser.add_argument("--device", default=None, help='Override compute device, e.g. "cuda:0" or "cpu".')
    parser.add_argument("--init-checkpoint", default=None, help="Load matching model weights before training.")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze deterministic backbone parameters.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_root = Path(args.data_root or config["data"]["root"])
    output_dir = Path(args.output_dir or config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    _set_seed(int(config.get("seed", 42)))
    variant = args.variant or config["model"].get("variant", "full")
    _apply_variant(config, variant)
    if args.init_checkpoint:
        config.setdefault("training", {})["init_checkpoint"] = args.init_checkpoint
    if args.freeze_backbone:
        config.setdefault("training", {})["freeze_backbone"] = True
    save_config(config, output_dir / "config.snapshot.json")

    train_split = list_statehaze_pairs(data_root, "train")
    test_split = list_statehaze_pairs(data_root, str(config["data"].get("eval_split", "test")))
    max_train = config["data"].get("max_train_samples")
    max_test = config["data"].get("max_test_samples")
    if max_train:
        train_split = train_split[: int(max_train)]
    if max_test:
        test_split = test_split[: int(max_test)]
    image_size = int(config["data"].get("image_size", 128))
    train_loader = DataLoader(
        StateHazeDataset(train_split, image_size=image_size, training=True),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["data"].get("num_workers", 2)),
        pin_memory=True,
    )
    test_loader = DataLoader(
        StateHazeDataset(test_split, image_size=None, training=False),
        batch_size=1,
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 2)),
        pin_memory=True,
    )

    device = _resolve_device(args.device)
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
    if args.init_checkpoint:
        loaded = _load_matching_weights(model, args.init_checkpoint, device)
        print(
            "initialized_from="
            f"{args.init_checkpoint} loaded={loaded['loaded']} skipped={loaded['skipped']}"
        )
    if bool(config["training"].get("freeze_backbone", False)):
        _freeze_backbone(model)
        print("frozen=backbone")
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters remain after applying freeze options.")
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(config["training"].get("amp", True)) and device.type == "cuda")
    start_epoch = 0
    monitor_metric = str(config["training"].get("save_best_by", "psnr")).lower()
    if monitor_metric not in {"psnr", "ssim"}:
        raise ValueError("training.save_best_by must be either 'psnr' or 'ssim'")
    best_metric = float("-inf")
    resume_path = _resolve_resume_path(args.resume, output_dir, config)
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_metric = float(checkpoint.get("best_metric", checkpoint.get("best_psnr", float("-inf"))))
        print(
            f"resume_from={resume_path} start_epoch={start_epoch} "
            f"best_{monitor_metric}={best_metric:.6f}"
        )

    log_path = output_dir / "train_log.csv"
    fieldnames = ["epoch", "train_loss", "test_psnr", "test_ssim", "test_l1"]
    if not log_path.exists():
        with log_path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fieldnames).writeheader()

    weights = {key: float(value) for key, value in config.get("loss", {}).items()}
    for epoch in range(start_epoch, int(config["training"]["epochs"])):
        train_loss = _train_one_epoch(model, train_loader, optimizer, scaler, weights, device)
        test_metrics = _validate(model, test_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_psnr": test_metrics["psnr"],
            "test_ssim": test_metrics["ssim"],
            "test_l1": test_metrics["l1"],
        }
        with log_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fieldnames).writerow(row)

        current_metric = float(test_metrics[monitor_metric])
        is_best = current_metric > best_metric
        best_metric = max(best_metric, current_metric)
        _save_checkpoint(
            output_dir / "checkpoints" / "latest.pth",
            model,
            optimizer,
            scaler,
            epoch,
            best_metric,
            monitor_metric,
            config,
        )
        if is_best:
            _save_checkpoint(
                output_dir / "checkpoints" / "best.pth",
                model,
                optimizer,
                scaler,
                epoch,
                best_metric,
                monitor_metric,
                config,
            )
            _save_preview(model, test_loader, device, output_dir / "samples" / f"epoch_{epoch:03d}.png")
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} "
            f"test_psnr={test_metrics['psnr']:.4f} test_ssim={test_metrics['ssim']:.6f} "
            f"test_l1={test_metrics['l1']:.6f} best_{monitor_metric}={best_metric:.6f}"
        )


def _train_one_epoch(
    model: PFDiffDehaze,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    weights: Dict[str, float],
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        hazy = batch["hazy"].to(device, non_blocking=True)
        clear = batch["clear"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            losses = model.training_losses(hazy, clear, weights)
            loss = losses["loss_total"]
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach().cpu())
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def _validate(model: PFDiffDehaze, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    psnrs = []
    ssims = []
    l1s = []
    for batch in loader:
        hazy = batch["hazy"].to(device, non_blocking=True)
        clear = batch["clear"].to(device, non_blocking=True)
        pred = model(hazy)
        mse = torch.mean((pred - clear) ** 2, dim=(1, 2, 3)).clamp_min(1e-12)
        psnrs.extend((-10.0 * torch.log10(mse)).detach().cpu().tolist())
        l1s.extend(torch.mean(torch.abs(pred - clear), dim=(1, 2, 3)).detach().cpu().tolist())
        ssims.extend(_global_ssim_per_image(pred, clear).detach().cpu().tolist())
    return {
        "psnr": sum(psnrs) / len(psnrs),
        "ssim": sum(ssims) / len(ssims),
        "l1": sum(l1s) / len(l1s),
    }


@torch.no_grad()
def _save_preview(model: PFDiffDehaze, loader: DataLoader, device: torch.device, path: Path) -> None:
    batch = next(iter(loader))
    hazy = batch["hazy"].to(device)
    pred = model(hazy)
    save_tensor_image(pred[0], path)


def _save_checkpoint(
    path: Path,
    model: PFDiffDehaze,
    optimizer,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    best_metric: float,
    metric_name: str,
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "epoch": epoch,
            "best_metric": best_metric,
            "best_psnr": best_metric if metric_name == "psnr" else None,
            "save_best_by": metric_name,
            "config": config,
        },
        path,
    )


def _resolve_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_resume_path(resume_arg: Optional[str], output_dir: Path, config: dict) -> Optional[Path]:
    if resume_arg:
        if resume_arg == "latest":
            candidate = output_dir / "checkpoints" / "latest.pth"
        else:
            candidate = Path(resume_arg)
        if not candidate.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {candidate}")
        return candidate
    if bool(config.get("training", {}).get("auto_resume", False)):
        candidate = output_dir / "checkpoints" / "latest.pth"
        if candidate.exists():
            return candidate
    return None


def _global_ssim_per_image(pred: torch.Tensor, clear: torch.Tensor) -> torch.Tensor:
    # Lightweight global SSIM approximation used for checkpoint selection.
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


def _load_matching_weights(model: PFDiffDehaze, checkpoint_path: str, device: torch.device) -> Dict[str, int]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    source_state = checkpoint["model"] if "model" in checkpoint else checkpoint
    target_state = model.state_dict()
    matched = {}
    skipped = 0
    for key, value in source_state.items():
        if key in target_state and target_state[key].shape == value.shape:
            matched[key] = value
        else:
            skipped += 1
    target_state.update(matched)
    model.load_state_dict(target_state)
    return {"loaded": len(matched), "skipped": skipped}


def _freeze_backbone(model: PFDiffDehaze) -> None:
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    model.backbone.eval()


def _apply_variant(config: dict, variant: str) -> None:
    model_cfg = config["model"]
    model_cfg["variant"] = variant
    model_cfg["use_diffusion"] = variant in {"diffusion", "physics", "frequency", "full"}
    model_cfg["use_physics"] = variant in {"physics", "full"}
    model_cfg["use_frequency"] = variant in {"frequency", "full"}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
