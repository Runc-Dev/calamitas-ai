"""Tests for SegmentationMetrics edge cases (review finding #12)."""

from __future__ import annotations

import math

import numpy as np

from afetsonar.evaluation.metrics import SegmentationMetrics


def test_empty_confusion_returns_zero_not_nan():
    """With no accumulated pixels every summary metric must be a finite
    number (0.0), never NaN — NaN breaks JSON reports downstream."""
    metrics = SegmentationMetrics(num_classes=6)
    scores = metrics.compute()
    for key in ("miou", "miou_no_bg", "mf1", "accuracy"):
        assert math.isfinite(scores[key]), f"{key} is not finite"
        assert scores[key] == 0.0


def test_absent_classes_excluded_from_mean():
    """Classes never seen in preds or targets must not drag the mean."""
    metrics = SegmentationMetrics(num_classes=3)
    # Only class 1 present, predicted perfectly; classes 0 and 2 absent.
    preds = np.ones((1, 8, 8), dtype=np.int64)
    targets = np.ones((1, 8, 8), dtype=np.int64)
    metrics.update(preds, targets)
    scores = metrics.compute()

    assert scores["miou_no_bg"] == 1.0  # mean over present classes only
    assert math.isnan(scores["iou_per_class"][2])  # absent stays NaN in detail


def test_perfect_and_wrong_predictions():
    metrics = SegmentationMetrics(num_classes=2)
    preds = np.array([[[1, 1], [0, 0]]], dtype=np.int64)
    targets = np.array([[[1, 0], [0, 0]]], dtype=np.int64)
    metrics.update(preds, targets)
    scores = metrics.compute()
    # class1: tp=1 fp=1 fn=0 -> IoU 0.5 ; class0: tp=2 fp=0 fn=1 -> 2/3
    assert abs(scores["iou_per_class"][1] - 0.5) < 1e-9
    assert abs(scores["iou_per_class"][0] - 2 / 3) < 1e-9
