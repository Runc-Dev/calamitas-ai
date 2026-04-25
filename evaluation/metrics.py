"""Streaming segmentation and classification metrics.

Both classes keep internal accumulators so they can be fed batches in a loop
and produce scalar scores only at the end.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch


class SegmentationMetrics:
    """Streaming confusion matrix for multi-class segmentation.

    Example
    -------
    >>> metrics = SegmentationMetrics(num_classes=6)
    >>> for batch in loader:
    ...     preds = model(batch).argmax(dim=1)
    ...     metrics.update(preds, batch["mask"])
    >>> scores = metrics.compute()
    """

    def __init__(self, num_classes: int, ignore_index: Optional[int] = None) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        self.confusion = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )

    @torch.no_grad()
    def update(
        self,
        preds: "torch.Tensor | np.ndarray",
        targets: "torch.Tensor | np.ndarray",
    ) -> None:
        """Accumulate a batch of predictions and targets.

        Both arguments must be long-typed ``[B, H, W]`` (or flattened).
        """
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        preds = preds.flatten()
        targets = targets.flatten()

        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            preds = preds[mask]
            targets = targets[mask]

        valid = (targets >= 0) & (targets < self.num_classes)
        preds = preds[valid]
        targets = targets[valid]

        idx = self.num_classes * targets + preds
        binc = np.bincount(idx, minlength=self.num_classes ** 2)
        self.confusion += binc.reshape(self.num_classes, self.num_classes)

    def compute(self) -> Dict[str, Any]:
        """Finalize and return mIoU / F1 / pixel accuracy statistics."""
        cm = self.confusion.astype(np.float64)

        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp

        iou = tp / np.maximum(tp + fp + fn, 1.0)
        iou[(tp + fp + fn) == 0] = float("nan")

        precision = tp / np.maximum(tp + fp, 1.0)
        recall = tp / np.maximum(tp + fn, 1.0)
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-8)
        f1[(tp + fp + fn) == 0] = float("nan")

        accuracy = tp.sum() / max(cm.sum(), 1.0)
        miou = float(np.nanmean(iou))
        miou_no_bg = float(np.nanmean(iou[1:]))
        mf1 = float(np.nanmean(f1))

        return {
            "miou": miou,
            "miou_no_bg": miou_no_bg,
            "iou_per_class": iou.tolist(),
            "accuracy": float(accuracy),
            "f1_per_class": f1.tolist(),
            "mf1": mf1,
        }


class ClassificationMetrics:
    """Accuracy + balanced accuracy for image-level classification."""

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        self.correct = 0
        self.total = 0
        self.per_class_correct = np.zeros(self.num_classes, dtype=np.int64)
        self.per_class_total = np.zeros(self.num_classes, dtype=np.int64)

    @torch.no_grad()
    def update(
        self,
        preds: "torch.Tensor | np.ndarray",
        targets: "torch.Tensor | np.ndarray",
    ) -> None:
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        self.correct += int((preds == targets).sum())
        self.total += len(targets)
        for c in range(self.num_classes):
            mask = targets == c
            self.per_class_correct[c] += int((preds[mask] == c).sum())
            self.per_class_total[c] += int(mask.sum())

    def compute(self) -> Dict[str, Any]:
        accuracy = self.correct / max(self.total, 1)
        per_class_acc = self.per_class_correct / np.maximum(self.per_class_total, 1)
        return {
            "accuracy": float(accuracy),
            "per_class_accuracy": per_class_acc.tolist(),
            "balanced_accuracy": float(np.nanmean(per_class_acc)),
        }


__all__ = ["ClassificationMetrics", "SegmentationMetrics"]
