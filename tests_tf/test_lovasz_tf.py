"""Behavioural sanity tests for the static-shape TF Lovász port."""

from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from afetsonar_tf.losses import lovasz_softmax_tf  # noqa: E402


def _one_hot_logits(labels: np.ndarray, num_classes: int,
                    scale: float) -> tf.Tensor:
    """Logits strongly favouring the given labels (scale>0) or a wrong
    class (scale<0)."""
    onehot = np.eye(num_classes, dtype=np.float32)[labels]
    return tf.constant(scale * onehot)


def test_perfect_prediction_near_zero():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 6, (2, 32, 32))
    logits = _one_hot_logits(labels, 6, scale=50.0)
    loss = float(lovasz_softmax_tf(logits, labels))
    assert loss < 1e-3


def test_wrong_prediction_is_worse():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 6, (2, 32, 32))
    good = float(lovasz_softmax_tf(_one_hot_logits(labels, 6, 50.0), labels))
    shifted = (labels + 1) % 6
    bad = float(lovasz_softmax_tf(_one_hot_logits(shifted, 6, 50.0), labels))
    assert bad > good + 0.5


def test_absent_classes_do_not_contribute():
    """Only classes 0/1 present: weights of absent classes must not
    change the loss (presence masking, mirrors torch 'present' mode)."""
    rng = np.random.default_rng(2)
    labels = rng.integers(0, 2, (1, 32, 32))
    logits = tf.constant(rng.standard_normal((1, 32, 32, 6)).astype(np.float32))

    w1 = [1.0, 1.0, 99.0, 99.0, 99.0, 99.0]
    w2 = [1.0, 1.0, 0.1, 0.1, 0.1, 0.1]
    l1 = float(lovasz_softmax_tf(logits, labels, class_weights=w1))
    l2 = float(lovasz_softmax_tf(logits, labels, class_weights=w2))
    assert abs(l1 - l2) < 1e-6
