"""Phase 3 student distillation training script.

Trains StudentSiameseSegformer (B0) to match the teacher (B3) using a
5-component knowledge distillation loss.

Usage::

    python scripts/train_student.py \\
        --teacher-ckpt checkpoints/teacher/teacher_v4_best_ema.pth \\
        --config configs/student.yaml \\
        --output-dir checkpoints/student
"""

from __future__ import annotations

import argparse
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
from afetsonar.losses import KnowledgeDistillationLoss
from afetsonar.models import ModelEMA, SiameseTeacherSegformerV3, StudentSiameseSegformer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train AFETSONAR Student (Phase 3 KD)")
    p.add_argument("--teacher-ckpt", required=True)
    p.add_argument("--config", default="configs/student.yaml")
    p.add_argument("--data-dir", default="data/xbd")
    p.add_argument("--output-dir", default="checkpoints/student")
    p.add_argument("--train-csv", default=None)
    p.add_argument("--val-csv", default=None)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    cfg = DefaultConfig.from_yaml(args.config) if Path(args.config).exists() else DefaultConfig()
    cfg.total_epochs = args.epochs
    cfg.batch_size_student = args.batch_size

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load teacher (frozen)
    print(f"Loading teacher: {args.teacher_ckpt}")
    teacher = SiameseTeacherSegformerV3(
        num_damage_classes=cfg.num_classes, pretrained=False
    ).to(device)
    ckpt = torch.load(args.teacher_ckpt, map_location=device)
    teacher.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Student
    student = StudentSiameseSegformer(
        num_damage_classes=cfg.num_classes,
        num_disaster_classes=cfg.num_disaster_classes,
    ).to(device)
    ema = ModelEMA(student, decay=cfg.ema_decay)

    criterion = KnowledgeDistillationLoss(
        num_classes=cfg.num_classes,
        class_weights=cfg.class_weights,
        temperature=cfg.kd_temperature,
    )
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    # Data
    import os
    train_csv = args.train_csv or os.path.join(args.data_dir, "splits", "train.csv")
    val_csv = args.val_csv or os.path.join(args.data_dir, "splits", "val.csv")

    train_ds = XBDDatasetV2(train_csv, mode="teacher",
                             augmentation=get_train_augmentation_v2(cfg.image_size),
                             image_size=cfg.image_size)
    val_ds = XBDDatasetV2(val_csv, mode="teacher",
                           augmentation=get_val_augmentation_v2(cfg.image_size),
                           image_size=cfg.image_size, building_aware_crop=False)

    weights = train_ds.get_sample_weights()
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    best_miou = 0.0
    print(f"Student: {student.num_parameters()/1e6:.1f}M params | device={device}")

    for epoch in range(args.epochs):
        student.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            with torch.no_grad():
                teacher_out = teacher(images)

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                student_out = student(images)
                loss_dict = criterion(student_out, teacher_out, {"damage_mask": masks})
                loss = loss_dict["total"]

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(student)

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validation with EMA weights
        backup = ema.apply_to(student)
        student.eval()
        metrics = SegmentationMetrics(num_classes=cfg.num_classes)
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                msks = batch["mask"].to(device)
                out = student(imgs)
                logits = out["damage_logits"]
                metrics.update(logits.argmax(dim=1), msks)
        ema.restore(student, backup)
        scores = metrics.compute()
        miou = scores["miou_no_bg"]

        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"loss={total_loss/n_batches:.4f} | "
            f"mIoU_no_bg={miou:.3f} | mF1={scores['mf1']:.3f}"
        )

        if miou > best_miou:
            best_miou = miou
            save_path = out_dir / "student_v1_best_ema.pth"
            torch.save(
                {"epoch": epoch + 1, "model_state_dict": student.state_dict(),
                 "ema_shadow": ema.shadow, "val_miou_no_bg": miou},
                save_path,
            )
            print(f"  ✓ Best student saved → {save_path}")

    print(f"\nDistillation complete. Best student mIoU_no_bg: {best_miou:.3f}")


if __name__ == "__main__":
    main()
