"""TF ports of the AFETSONAR losses (parity-tested against PyTorch)."""

from afetsonar_tf.losses.lovasz_tf import lovasz_softmax_tf
from afetsonar_tf.losses.combo_tf import (
    combo_damage_loss_tf,
    deep_supervision_loss_tf,
    dice_loss_tf,
    focal_loss_tf,
    teacher_loss_v3_tf,
)

__all__ = [
    "lovasz_softmax_tf",
    "focal_loss_tf",
    "dice_loss_tf",
    "combo_damage_loss_tf",
    "deep_supervision_loss_tf",
    "teacher_loss_v3_tf",
]
