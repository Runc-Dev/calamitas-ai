"""Segmentation and classification metrics for AFETSONAR evaluation.

Both ``SegmentationMetrics`` and ``ClassificationMetrics`` are streaming —
call ``update()`` per batch, then ``compute()`` at the end of an epoch.
They accept both NumPy arrays and PyTorch tensors.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch


class SegmentationMetrics:
    """Streaming confusion-matrix based segmentation metrics.

    Computes mIoU, mF1, per-class IoU/F1, and pixel accuracy from a running
    confusion matrix accumulated across batches.

    Args:
        num_classes: Total number of classes (including background).
        ignore_index: Label value to exclude (e.g. ``-100`` for ignored pixels).

    Example:
        >>> metrics = SegmentationMetrics(num_classes=6)
        >>> for preds, targets in eval_loop:
        ...     metrics.update(preds.argmax(1), targets)
        >>> scores = metrics.compute()
        >>> print(f"mIoU_no_bg: {scores['miou_no_bg']:.3f}")
    """

    def __init__(self, num_classes: int, ignore_index: Optional[int] = None) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        """Clear accumulated confusion matrix."""
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    @torch.no_grad()
    def update(self, preds: "np.ndarray | torch.Tensor", targets: "np.ndarray | torch.Tensor") -> None:
        """Accumulate predictions into the confusion matrix.

        Args:
            preds: Predicted class indices ``(B, H, W)`` — argmax should
                already be applied.
            targets: Ground-truth class indices ``(B, H, W)``.
        """
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        preds = preds.flatten()
        targets = targets.flatten()

        if self.ignore_index is not None:
            mask = targets != self.ignore_index
            preds, targets = preds[mask], targets[mask]

        valid = (targets >= 0) & (targets < self.num_classes)
        preds, targets = preds[valid], targets[valid]

        idx = self.num_classes * targets + preds
        self.confusion += np.bincount(idx, minlength=self.num_classes ** 2).reshape(
            self.num_classes, self.num_classes
        )

    def compute(self) -> Dict[str, object]:
        """Compute all metrics from the accumulated confusion matrix.

        Returns:
            Dict with keys:

            - ``"miou"`` — mean IoU (all classes).
            - ``"miou_no_bg"`` — mean IoU excluding class 0 (background).
            - ``"iou_per_class"`` — list of per-class IoU values.
            - ``"accuracy"`` — overall pixel accuracy.
            - ``"f1_per_class"`` — list of per-class F1 values.
            - ``"mf1"`` — mean F1 (all classes).
        """
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

        return {
            "miou": float(np.nanmean(iou)),
            "miou_no_bg": float(np.nanmean(iou[1:])),
            "iou_per_class": iou.tolist(),
            "accuracy": float(accuracy),
            "f1_per_class": f1.tolist(),
            "mf1": float(np.nanmean(f1)),
        }


class ClassificationMetrics:
    """Streaming accuracy metrics for image-level classification.

    Args:
        num_classes: Number of class labels.

    Example:
        >>> metrics = ClassificationMetrics(num_classes=5)
        >>> for preds, targets in eval_loop:
        ...     metrics.update(preds.argmax(1), targets)
        >>> scores = metrics.compute()
    """

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        """Clear accumulated statistics."""
        self.correct: int = 0
        self.total: int = 0
        self.per_class_correct = np.zeros(self.num_classes, dtype=np.int64)
        self.per_class_total = np.zeros(self.num_classes, dtype=np.int64)

    @torch.no_grad()
    def update(
        self,
        preds: "np.ndarray | torch.Tensor",
        targets: "np.ndarray | torch.Tensor",
    ) -> None:
        """Accumulate batch predictions.

        Args:
            preds: Predicted class indices ``(B,)`` (argmax already applied).
            targets: Ground-truth class indices ``(B,)``.
        """
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        self.correct += int((preds == targets).sum())
        self.total += len(targets)
        for c in range(self.num_classes):
            m = targets == c
            self.per_class_correct[c] += int((preds[m] == c).sum())
            self.per_class_total[c] += int(m.sum())

    def compute(self) -> Dict[str, object]:
        """Compute accuracy metrics.

        Returns:
            Dict with ``"accuracy"``, ``"per_class_accuracy"`` (list),
            and ``"balanced_accuracy"``.
        """
        accuracy = self.correct / max(self.total, 1)
        pca = self.per_class_correct / np.maximum(self.per_class_total, 1)
        return {
            "accuracy": float(accuracy),
            "per_class_accuracy": pca.tolist(),
            "balanced_accuracy": float(np.nanmean(pca)),
        }
