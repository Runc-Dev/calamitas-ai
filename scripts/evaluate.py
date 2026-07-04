"""Evaluation script — computes mIoU, mF1, per-class metrics.

Usage::

    python scripts/evaluate.py \\
        --model checkpoints/student/student_v1_best_ema.pth \\
        --test-csv data/xbd/splits/test.csv \\
        --output results/eval_results.json

Tier-1 SoTA options (Izmailov 2018 / TTA literature)::

    --tta                      8-transform geometric TTA
    --tta-scales 0.75 1.0 1.25 add multi-scale averaging
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from afetsonar.config import CLASS_NAMES, DefaultConfig
from afetsonar.data import XBDDatasetV2, get_val_augmentation_v2
from afetsonar.evaluation import SegmentationMetrics, tta_forward
from afetsonar.pipeline import AfetsonarPipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AFETSONAR evaluation")
    p.add_argument("--model",    required=True)
    p.add_argument("--test-csv", required=True)
    p.add_argument("--output",   default="results/eval_results.json")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device",   default="auto")
    p.add_argument("--image-size", type=int, default=None,
                   help="Defaults to the model's native resolution "
                        "(teacher: 768, student: 512)")
    p.add_argument("--tta", action="store_true",
                   help="Enable 8-transform geometric test-time augmentation")
    p.add_argument("--tta-scales", type=float, nargs="+", default=[1.0],
                   help="Multi-scale TTA factors, e.g. --tta-scales 0.75 1.0 1.25")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Model — architecture (teacher/student) auto-detected from checkpoint;
    # the pipeline also sets its native inference resolution.
    pipeline = AfetsonarPipeline(args.model, device=args.device)
    model, device = pipeline.model, pipeline.device

    cfg = DefaultConfig()
    cfg.image_size = args.image_size or pipeline.config.image_size
    print(f"Evaluation resolution: {cfg.image_size}px")

    # Dataset
    ds = XBDDatasetV2(
        args.test_csv, mode="teacher",
        augmentation=get_val_augmentation_v2(cfg.image_size),
        image_size=cfg.image_size, building_aware_crop=False,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    metrics = SegmentationMetrics(num_classes=cfg.num_classes)
    t0 = time.time()

    if args.tta:
        n_infer = 8 * len(args.tta_scales)
        print(f"TTA enabled: 8 transforms x {len(args.tta_scales)} scale(s) "
              f"= {n_infer} inferences/image")

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            msks = batch["mask"].to(device)
            if args.tta:
                probs = tta_forward(model, imgs, n_augmentations=8,
                                    scales=tuple(args.tta_scales))
                preds = probs.argmax(dim=1)
            else:
                logits = model(imgs)["damage_logits"]
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                preds = logits.argmax(dim=1)
            metrics.update(preds, msks)

    elapsed = time.time() - t0
    scores = metrics.compute()

    # Pretty print
    print("\n" + "="*60)
    print("AFETSONAR EVALUATION RESULTS")
    print("="*60)
    print(f"mIoU (all)    : {scores['miou']:.4f}")
    print(f"mIoU (no bg)  : {scores['miou_no_bg']:.4f}")
    print(f"mF1           : {scores['mf1']:.4f}")
    print(f"Accuracy      : {scores['accuracy']:.4f}")
    print(f"\nPer-class IoU:")
    for i, iou in enumerate(scores["iou_per_class"]):
        name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"cls_{i}"
        print(f"  {name:<15}: {iou:.4f}")
    print(f"\nTotal time: {elapsed:.1f}s | {len(ds)/elapsed:.1f} img/s")

    # Save JSON
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results = {
        "metrics": scores,
        "timing": {"total_s": elapsed, "img_per_s": len(ds)/elapsed},
        "model": args.model,
        "tta": {"enabled": args.tta, "scales": args.tta_scales},
        "image_size": cfg.image_size,
    }
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()
