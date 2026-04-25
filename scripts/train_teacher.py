"""Phase 2 teacher training script.

Trains the SiameseTeacherSegformerV3 model on xBD data with:
- Lovász+Dice+Focal combo loss with deep supervision
- Cosine warm restarts (3 cycles)
- EMA model averaging (decay 0.999)
- WeightedRandomSampler for class balance
- Gradient checkpointing (optional, saves VRAM)

Usage::

    python scripts/train_teacher.py \\
        --config configs/teacher.yaml \\
        --data-dir data/xbd \\
        --output-dir checkpoints/teacher
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from afetsonar.config import DefaultConfig
from afetsonar.data import XBDDatasetV2, get_train_augmentation_v2, get_val_augmentation_v2
from afetsonar.evaluation import SegmentationMetrics
from afetsonar.losses import TeacherLossV3, derive_building_mask, derive_change_mask
from afetsonar.models import ModelEMA, SiameseTeacherSegformerV3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train AFETSONAR Teacher (Phase 2)")
    p.add_argument("--config", default="configs/teacher.yaml")
    p.add_argument("--data-dir", default="data/xbd")
    p.add_argument("--output-dir", default="checkpoints/teacher")
    p.add_argument("--train-csv", default=None)
    p.add_argument("--val-csv", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--resume", default=None, help="Checkpoint to resume from")
    p.add_argument("--grad-ckpt", action="store_true", help="Enable gradient checkpointing")
    return p.parse_args()


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: TeacherLossV3,
    optimizer: torch.optim.Optimizer,
    ema: ModelEMA,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
) -> dict:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        disaster_idx = batch["disaster_idx"].to(device)

        change_masks = derive_change_mask(masks)

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            outputs = model(images)
            targets = {
                "damage_mask": masks,
                "change_mask": change_masks,
                "disaster_idx": disaster_idx,
            }
            loss_dict = criterion(outputs, targets)
            loss = loss_dict["total"]

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)

        total_loss += loss.item()
        n_batches += 1

    return {"loss": total_loss / max(n_batches, 1)}


@torch.no_grad()
def val_epoch(
    model: nn.Module,
    loader: DataLoader,
    ema: ModelEMA,
    device: torch.device,
    num_classes: int,
) -> dict:
    backup = ema.apply_to(model)
    model.eval()
    metrics = SegmentationMetrics(num_classes=num_classes)

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            outputs = model(images)
        logits = outputs["damage_logits"]
        if isinstance(logits, list):
            logits = logits[0]
        preds = logits.argmax(dim=1)
        metrics.update(preds, masks)

    ema.restore(model, backup)
    scores = metrics.compute()
    return scores


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    # Config
    cfg = DefaultConfig.from_yaml(args.config) if Path(args.config).exists() else DefaultConfig()
    if args.epochs:
        cfg.total_epochs = args.epochs
    if args.batch_size:
        cfg.batch_size_teacher = args.batch_size

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data
    train_csv = args.train_csv or os.path.join(args.data_dir, "splits", "train.csv")
    val_csv = args.val_csv or os.path.join(args.data_dir, "splits", "val.csv")

    train_ds = XBDDatasetV2(
        train_csv, mode="teacher",
        augmentation=get_train_augmentation_v2(cfg.image_size, "teacher"),
        image_size=cfg.image_size,
    )
    val_ds = XBDDatasetV2(
        val_csv, mode="teacher",
        augmentation=get_val_augmentation_v2(cfg.image_size, "teacher"),
        image_size=cfg.image_size,
        building_aware_crop=False,
    )

    weights = train_ds.get_sample_weights()
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size_teacher, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size_teacher, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = SiameseTeacherSegformerV3(
        num_damage_classes=cfg.num_classes,
        num_disaster_classes=cfg.num_disaster_classes,
        use_deep_supervision=True,
    ).to(device)

    if args.grad_ckpt:
        model.enable_gradient_checkpointing()

    ema = ModelEMA(model, decay=cfg.ema_decay)
    criterion = TeacherLossV3(
        num_damage_classes=cfg.num_classes,
        damage_class_weights=cfg.class_weights,
        use_deep_supervision=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_peaks[0], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.total_epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    # Resume
    start_epoch = 0
    best_miou = 0.0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        start_epoch = ckpt.get("epoch", 0)
        best_miou = ckpt.get("val_miou_no_bg", 0.0)
        print(f"Resumed from epoch {start_epoch}, best mIoU={best_miou:.3f}")

    print(f"Training on {device} | {model.num_parameters()/1e6:.1f}M params | {len(train_ds)} train samples")

    for epoch in range(start_epoch, cfg.total_epochs):
        t0 = time.time()
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, ema, device, scaler)
        val_metrics = val_epoch(model, val_loader, ema, device, cfg.num_classes)
        scheduler.step()

        miou = val_metrics["miou_no_bg"]
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch+1:3d}/{cfg.total_epochs} | "
            f"loss={train_metrics['loss']:.4f} | "
            f"mIoU_no_bg={miou:.3f} | mF1={val_metrics['mf1']:.3f} | "
            f"time={elapsed:.0f}s"
        )

        if miou > best_miou:
            best_miou = miou
            save_path = out_dir / "teacher_v4_best_ema.pth"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "ema_shadow": ema.shadow,
                    "val_miou_no_bg": miou,
                    "val_mf1": val_metrics["mf1"],
                    "config": cfg.__dict__,
                },
                save_path,
            )
            print(f"  ✓ Best model saved → {save_path}  (mIoU_no_bg={miou:.3f})")

    print(f"\nTraining complete. Best mIoU_no_bg: {best_miou:.3f}")


if __name__ == "__main__":
    main()
