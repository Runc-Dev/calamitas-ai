"""TF ports of Focal / Dice / Combo / DeepSupervision / TeacherLossV3.

Line-for-line semantic ports of ``afetsonar/losses/combo.py`` —
constants and reductions must not drift (golden-value tests pin them):
Combo = 0.35*Lovász + 0.35*Dice + 0.30*Focal; Dice smooth=1.0,
background excluded; Focal gamma=2.0 with per-class alpha;
DeepSupervision weights [1.0, 0.4, 0.3, 0.2];
TeacherV3 = 0.70*damage + 0.20*CE(change) + 0.10*CE(disaster).

All functions take NHWC logits and int labels; all math is float32.
``ignore_index`` is not supported (xBD masks are always 0–5).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import tensorflow as tf

from afetsonar_tf.losses.lovasz_tf import lovasz_softmax_tf

TensorOrList = Union[tf.Tensor, List[tf.Tensor]]


def focal_loss_tf(
    logits: tf.Tensor,
    targets: tf.Tensor,
    gamma: float = 2.0,
    alpha: Optional[Sequence[float]] = None,
) -> tf.Tensor:
    """Focal loss (Lin et al. 2017) — mirrors ``FocalLoss.forward``."""
    logits = tf.cast(logits, tf.float32)
    targets = tf.cast(targets, tf.int32)
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=targets, logits=logits
    )                                                    # (B, H, W)
    p_t = tf.exp(-ce)
    weight = tf.pow(1.0 - p_t, gamma)
    if alpha is not None:
        alpha_t = tf.gather(
            tf.constant(list(alpha), dtype=tf.float32), targets
        )
        weight = weight * alpha_t
    return tf.reduce_mean(weight * ce)


def dice_loss_tf(
    logits: tf.Tensor,
    targets: tf.Tensor,
    num_classes: int,
    smooth: float = 1.0,
    class_weights: Optional[Sequence[float]] = None,
    exclude_background: bool = True,
) -> tf.Tensor:
    """Soft multi-class Dice — mirrors ``DiceLoss.forward``.

    Absent classes contribute dice=1 (zero loss) via the smoothing term,
    but their weight stays in the denominator — exactly like torch.
    """
    probs = tf.nn.softmax(tf.cast(logits, tf.float32), axis=-1)
    onehot = tf.one_hot(
        tf.cast(targets, tf.int32), num_classes, dtype=tf.float32
    )                                                    # (B, H, W, C)

    dims = (0, 1, 2)
    inter = tf.reduce_sum(probs * onehot, axis=dims)
    card = tf.reduce_sum(probs, axis=dims) + tf.reduce_sum(onehot, axis=dims)
    dice = (2.0 * inter + smooth) / (card + smooth)      # (C,)

    start = 1 if exclude_background else 0
    if class_weights is not None:
        if len(class_weights) != num_classes:
            raise ValueError(
                f"class_weights length {len(class_weights)} != "
                f"num_classes {num_classes}"
            )
        w = tf.constant(list(class_weights), dtype=tf.float32)[start:]
        return tf.reduce_sum((1.0 - dice[start:]) * w) / tf.reduce_sum(w)
    return tf.reduce_mean(1.0 - dice[start:])


def combo_damage_loss_tf(
    logits: tf.Tensor,
    targets: tf.Tensor,
    num_classes: int,
    class_weights: Optional[Sequence[float]] = None,
    focal_gamma: float = 2.0,
    lovasz_weight: float = 0.35,
    dice_weight: float = 0.35,
    focal_weight: float = 0.30,
) -> Dict[str, tf.Tensor]:
    """Weighted Lovász + Dice + Focal — mirrors ``ComboDamageLossV3``."""
    l_lovasz = lovasz_softmax_tf(logits, targets, class_weights=class_weights)
    l_dice = dice_loss_tf(
        logits, targets, num_classes, class_weights=class_weights
    )
    l_focal = focal_loss_tf(
        logits, targets, gamma=focal_gamma, alpha=class_weights
    )
    total = (lovasz_weight * l_lovasz
             + dice_weight * l_dice
             + focal_weight * l_focal)
    return {"total": total, "lovasz": l_lovasz, "dice": l_dice,
            "focal": l_focal}


def deep_supervision_loss_tf(
    logits_list: TensorOrList,
    targets: tf.Tensor,
    num_classes: int,
    class_weights: Optional[Sequence[float]] = None,
    aux_weights: Sequence[float] = (1.0, 0.4, 0.3, 0.2),
) -> Dict[str, tf.Tensor]:
    """Deep-supervision wrapper around the combo loss.

    ``logits_list[0]`` is the main decoder output; auxiliary logits are
    bilinearly resized to the target resolution when needed.
    """
    if not isinstance(logits_list, (list, tuple)):
        return combo_damage_loss_tf(
            logits_list, targets, num_classes, class_weights=class_weights
        )

    target_hw = tf.shape(targets)[1:3]
    total = None
    first: Optional[Dict[str, tf.Tensor]] = None
    for i, logits in enumerate(logits_list):
        if i >= len(aux_weights):
            break
        static_hw = logits.shape[1:3]
        if (static_hw[0] is None or static_hw[0] != targets.shape[1]
                or static_hw[1] != targets.shape[2]):
            logits = tf.image.resize(
                tf.cast(logits, tf.float32), target_hw, method="bilinear"
            )
        result = combo_damage_loss_tf(
            logits, targets, num_classes, class_weights=class_weights
        )
        weighted = aux_weights[i] * result["total"]
        total = weighted if total is None else total + weighted
        if first is None:
            first = result

    assert total is not None and first is not None
    return {"total": total, "lovasz": first["lovasz"],
            "dice": first["dice"], "focal": first["focal"]}


def teacher_loss_v3_tf(
    outputs: Dict[str, TensorOrList],
    targets: Dict[str, tf.Tensor],
    num_damage_classes: int = 6,
    damage_class_weights: Optional[Sequence[float]] = None,
    damage_weight: float = 0.70,
    change_weight: float = 0.20,
    disaster_weight: float = 0.10,
) -> Dict[str, tf.Tensor]:
    """Multi-task teacher loss — mirrors ``TeacherLossV3.forward``.

    Args:
        outputs: ``{"damage_logits": list[(B,H,W,6)], "change_logits":
            (B,H,W,2), "disaster_logits": (B,5)}`` (NHWC).
        targets: ``{"damage_mask": (B,H,W), "change_mask": (B,H,W),
            "disaster_idx": (B,)}`` integer tensors.
    """
    dmg = deep_supervision_loss_tf(
        outputs["damage_logits"], targets["damage_mask"],
        num_damage_classes, class_weights=damage_class_weights,
    )
    chg = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tf.cast(targets["change_mask"], tf.int32),
        logits=tf.cast(outputs["change_logits"], tf.float32),
    ))
    dis = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tf.cast(targets["disaster_idx"], tf.int32),
        logits=tf.cast(outputs["disaster_logits"], tf.float32),
    ))
    total = (damage_weight * dmg["total"]
             + change_weight * chg
             + disaster_weight * dis)
    return {
        "total": total,
        "damage": dmg["total"],
        "damage_lovasz": dmg["lovasz"],
        "damage_dice": dmg["dice"],
        "damage_focal": dmg["focal"],
        "change": chg,
        "disaster": dis,
    }
