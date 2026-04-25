"""Phase 1 building localization training script.

Trains LocalizerSegformer (SegFormer-B3, 2-class) on xBD data.
The trained encoder weights are transferred to the Phase 2 teacher.

Usage::

    python scripts/train_localizer.py \\
        --config configs/localizer.yaml \\
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
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from afetsonar.config import DefaultConfig
from afetsonar.data import XBDDatasetV2, get_train_augmentation_v2, get_val_augmentation_v2
from afetsonar.evaluation import SegmentationMetrics
from afetsonar.losses import LocalizationLoss, derive_building_mask
from afetsonar.models import LocalizerSegformer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train AFETSONAR Localizer (Phase 1)")
    p.add_argument("--config",      default="configs/localizer.yaml")
    p.add_argument("--data-dir",    default="data/xbd")
    p.add_argument("--output-dir",  default="checkpoints/teacher")
    p.add_argument("--train-csv",   default=None)
    p.add_argument("--val-csv",     default=None)
    p.add_argument("--epochs",      type=int, default=50)
    p.add_argument("--batch-size",  type=int, default=8)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--device",      default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else torch.device(args.device)

    cfg = DefaultConfig.from_yaml(args.config) if Path(args.config).exists() else DefaultConfig()
    image_size = getattr(cfg, "image_size", 512)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_csv = args.train_csv or os.path.join(args.data_dir, "splits", "train.csv")
    val_csv   = args.val_csv   or os.path.join(args.data_dir, "splits", "val.csv")

    # Use student mode (post-only, 3-ch) for Phase 1
    train_ds = XBDDatasetV2(
        train_csv, mode="student",
        augmentation=get_train_augmentation_v2(image_size, "student"),
        image_size=image_size,
    )
    val_ds = XBDDatasetV2(
        val_csv, mode="student",
        augmentation=get_val_augmentation_v2(image_size, "student"),
        image_size=image_size, building_aware_crop=False,
    )

    sampler = WeightedRandomSampler(
        train_ds.get_sample_weights(), len(train_ds), replacement=True
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,  num_workers=4, pin_memory=True)

    model = LocalizerSegformer().to(device)
    criterion = LocalizationLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_iou = 0.0
    print(f"Localizer: {model.num_parameters()/1e6:.1f}M params | device={device}")

    for epoch in range(args.epochs):
        # ── Train ──
        model.train()
        t0 = time.time()
        total_loss = 0.0
        for batch in train_loader:
            imgs  = batch["image"].to(device)
            masks = batch["mask"].to(device)
            bld_mask = derive_building_mask(masks)

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(imgs)
                loss_d = criterion(logits, bld_mask)
                loss   = loss_d["total"]

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        scheduler.step()

        # ── Val ──
        model.eval()
        metrics = SegmentationMetrics(num_classes=2)
        with torch.no_grad():
            for batch in val_loader:
                imgs  = batch["image"].to(device)
                masks = batch["mask"].to(device)
                bld   = derive_building_mask(masks)
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    logits = model(imgs)
                metrics.update(logits.argmax(1), bld)

        scores   = metrics.compute()
        bld_iou  = scores["iou_per_class"][1] if len(scores["iou_per_class"]) > 1 else 0.0
        elapsed  = time.time() - t0
        avg_loss = total_loss / max(len(train_loader), 1)

        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"loss={avg_loss:.4f} | building_IoU={bld_iou:.3f} | "
            f"time={elapsed:.0f}s"
        )

        if bld_iou > best_iou:
            best_iou = bld_iou
            save_path = out_dir / "localizer_v2_best.pth"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "encoder_state_dict": model.get_encoder_state_dict(),
                    "val_building_iou": bld_iou,
                },
                save_path,
            )
            print(f"  ✓ Best localizer saved → {save_path}  (building IoU={bld_iou:.3f})")

    print(f"\nPhase 1 complete.  Best building IoU: {best_iou:.3f}")
    print(f"Transfer encoder weights to teacher with --resume {out_dir}/localizer_v2_best.pth")


if __name__ == "__main__":
    main()
