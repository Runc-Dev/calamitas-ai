"""TF losses vs golden values computed by the PyTorch implementation.

The fixture inputs are stored NCHW (torch layout); TF functions take
NHWC, so logits are transposed here. Tolerances: closed-form losses
(CE/Dice/Focal) 5e-4 absolute; Lovász 1e-3 (sort ties may resolve
differently between frameworks).
"""

from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from afetsonar_tf.losses import (  # noqa: E402
    combo_damage_loss_tf,
    dice_loss_tf,
    focal_loss_tf,
    lovasz_softmax_tf,
    teacher_loss_v3_tf,
)


def _nhwc(x: np.ndarray) -> tf.Tensor:
    return tf.constant(np.transpose(x, (0, 2, 3, 1)))


def test_focal_matches_torch(golden_loss_data):
    inputs, values = golden_loss_data
    got = focal_loss_tf(
        _nhwc(inputs["logits_main"]), inputs["damage_mask"],
        gamma=2.0, alpha=inputs["class_weights"].tolist(),
    )
    assert abs(float(got) - values["focal"]) < 5e-4


def test_dice_matches_torch(golden_loss_data):
    inputs, values = golden_loss_data
    got = dice_loss_tf(
        _nhwc(inputs["logits_main"]), inputs["damage_mask"],
        num_classes=6, class_weights=inputs["class_weights"].tolist(),
    )
    assert abs(float(got) - values["dice"]) < 5e-4


def test_lovasz_matches_torch(golden_loss_data):
    inputs, values = golden_loss_data
    unweighted = lovasz_softmax_tf(
        _nhwc(inputs["logits_main"]), inputs["damage_mask"]
    )
    weighted = lovasz_softmax_tf(
        _nhwc(inputs["logits_main"]), inputs["damage_mask"],
        class_weights=inputs["class_weights"].tolist(),
    )
    assert abs(float(unweighted) - values["lovasz_unweighted"]) < 1e-3
    assert abs(float(weighted) - values["lovasz_weighted"]) < 1e-3


def test_combo_matches_torch(golden_loss_data):
    inputs, values = golden_loss_data
    got = combo_damage_loss_tf(
        _nhwc(inputs["logits_main"]), inputs["damage_mask"],
        num_classes=6, class_weights=inputs["class_weights"].tolist(),
    )
    assert abs(float(got["total"]) - values["combo_total"]) < 1.5e-3
    assert abs(float(got["dice"]) - values["combo_dice"]) < 5e-4
    assert abs(float(got["focal"]) - values["combo_focal"]) < 5e-4
    assert abs(float(got["lovasz"]) - values["combo_lovasz"]) < 1e-3


def test_teacher_loss_matches_torch(golden_loss_data):
    inputs, values = golden_loss_data
    outputs = {
        "damage_logits": [
            _nhwc(inputs["logits_main"]),
            _nhwc(inputs["logits_aux_0"]),
            _nhwc(inputs["logits_aux_1"]),
            _nhwc(inputs["logits_aux_2"]),
        ],
        "change_logits": _nhwc(inputs["change_logits"]),
        "disaster_logits": tf.constant(inputs["disaster_logits"]),
    }
    targets = {
        "damage_mask": tf.constant(inputs["damage_mask"]),
        "change_mask": tf.constant(inputs["change_mask"]),
        "disaster_idx": tf.constant(inputs["disaster_idx"]),
    }
    got = teacher_loss_v3_tf(
        outputs, targets,
        damage_class_weights=inputs["class_weights"].tolist(),
    )
    assert abs(float(got["change"]) - values["teacher_change"]) < 5e-4
    assert abs(float(got["disaster"]) - values["teacher_disaster"]) < 5e-4
    assert abs(float(got["damage"]) - values["teacher_damage"]) < 5e-3
    assert abs(float(got["total"]) - values["teacher_total"]) < 5e-3
