"""Combo damage loss and deep supervision wrapper (v3).

The final training objective for Phase 2 is a weighted combination of:

1. **Lovász-Softmax** (weight 0.35) — directly optimises mIoU.
2. **Dice Loss** (weight 0.35) — overlap metric, class-balanced.
3. **Focal Loss** (weight 0.30) — down-weights easy pixels, focuses on hard
   examples (especially minority classes).

``DeepSupervisionLoss`` wraps any ``base_loss`` to accept a list of logits
(one per decoder level) and returns a weighted sum.

``TeacherLossV3`` combines damage, binary-change, and disaster-type losses
for the multi-task teacher training objective.

References
----------
- Lin et al. 2017 — Focal Loss for Dense Object Detection (ICCV 2017).
- Milletari et al. 2016 — V-Net: Fully Convolutional Neural Networks for
  Volumetric Medical Image Segmentation (3DV 2016) — Dice loss.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from afetsonar.losses.lovasz import LovaszSoftmaxLoss


# ============================================================
# Focal Loss
# ============================================================

class FocalLoss(nn.Module):
    """Focal loss for multi-class segmentation.

    Args:
        gamma: Focusing parameter (default 2.0).
        alpha: Optional per-class weight tensor (length C).
        ignore_index: Label value excluded from loss.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[List[float]] = None,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        if alpha is not None:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha: Optional[torch.Tensor] = None  # type: ignore[assignment]

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none", ignore_index=self.ignore_index)
        p_t = torch.exp(-ce)
        weight = (1.0 - p_t) ** self.gamma
        if self.alpha is not None:
            # .to() guards callers that never moved this module to the GPU.
            weight = weight * self.alpha.to(ce.device)[targets.clamp(min=0)]
        valid = (targets != self.ignore_index).float()
        return (weight * ce * valid).sum() / valid.sum().clamp(min=1.0)


# ============================================================
# Dice Loss
# ============================================================

class DiceLoss(nn.Module):
    """Soft Dice loss for multi-class segmentation.

    Args:
        num_classes: Total number of classes including background.
        ignore_index: Label value excluded from loss.
        smooth: Laplace smoothing term.
        class_weights: Optional per-class weights.
        exclude_background: If ``True``, class 0 is excluded from the mean.
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = -100,
        smooth: float = 1.0,
        class_weights: Optional[List[float]] = None,
        exclude_background: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.exclude_background = exclude_background
        if class_weights is not None:
            if len(class_weights) != num_classes:
                raise ValueError(
                    f"class_weights length {len(class_weights)} != "
                    f"num_classes {num_classes}"
                )
            self.register_buffer(
                "class_weights", torch.tensor(class_weights, dtype=torch.float32)
            )
        else:
            self.class_weights: Optional[torch.Tensor] = None  # type: ignore[assignment]

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        valid = (targets != self.ignore_index).unsqueeze(1).float()
        t_clamped = targets.clamp(0, self.num_classes - 1)
        t_onehot = (
            F.one_hot(t_clamped, self.num_classes).permute(0, 3, 1, 2).float()
        )
        probs = probs * valid
        t_onehot = t_onehot * valid

        dims = (0, 2, 3)
        inter = (probs * t_onehot).sum(dims)
        card = probs.sum(dims) + t_onehot.sum(dims)
        dice = (2.0 * inter + self.smooth) / (card + self.smooth)

        start = 1 if self.exclude_background else 0
        if self.class_weights is not None:
            # .to() guards callers that never moved this module to the GPU.
            w = self.class_weights[start:].to(dice.device)
            return ((1.0 - dice[start:]) * w).sum() / w.sum()
        return (1.0 - dice[start:]).mean()


# ============================================================
# Combo Damage Loss v3
# ============================================================

class ComboDamageLossV3(nn.Module):
    """Weighted combination of Lovász + Dice + Focal losses.

    Args:
        num_classes: Number of damage classes.
        class_weights: Optional per-class weights (length ``num_classes``).
        focal_gamma: Focal loss focusing parameter.
        ignore_index: Ignored label value.
        lovasz_weight: Weight for Lovász term.
        dice_weight: Weight for Dice term.
        focal_weight: Weight for Focal term.
    """

    def __init__(
        self,
        num_classes: int,
        class_weights: Optional[List[float]] = None,
        focal_gamma: float = 2.0,
        ignore_index: int = -100,
        lovasz_weight: float = 0.35,
        dice_weight: float = 0.35,
        focal_weight: float = 0.30,
    ) -> None:
        super().__init__()
        self.lovasz = LovaszSoftmaxLoss(
            classes="present", ignore_index=ignore_index, class_weights=class_weights
        )
        self.dice = DiceLoss(
            num_classes=num_classes,
            class_weights=class_weights,
            ignore_index=ignore_index,
            exclude_background=True,
        )
        self.focal = FocalLoss(
            gamma=focal_gamma, alpha=class_weights, ignore_index=ignore_index
        )
        self.w_lovasz = lovasz_weight
        self.w_dice = dice_weight
        self.w_focal = focal_weight

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute combo loss.

        Args:
            logits: ``(B, C, H, W)`` raw logits.
            targets: ``(B, H, W)`` integer labels.

        Returns:
            Dict with keys ``total``, ``lovasz``, ``dice``, ``focal``.
        """
        l_lovasz = self.lovasz(logits, targets)
        l_dice = self.dice(logits, targets)
        l_focal = self.focal(logits, targets)
        total = self.w_lovasz * l_lovasz + self.w_dice * l_dice + self.w_focal * l_focal
        return {
            "total": total,
            "lovasz": l_lovasz.detach(),
            "dice": l_dice.detach(),
            "focal": l_focal.detach(),
        }


# ============================================================
# Deep Supervision Wrapper
# ============================================================

class DeepSupervisionLoss(nn.Module):
    """Wrap a base segmentation loss for deep supervision.

    Accepts a list of logits (one per decoder stage) and returns a weighted
    sum.  The primary decoder output is ``logits_list[0]`` and gets the
    highest weight.

    Args:
        base_loss: A loss module whose ``forward`` accepts ``(logits, targets)``
            and returns either a scalar or a dict with key ``"total"``.
        aux_weights: Weights for each logit in the list (should sum to ~1).
    """

    def __init__(
        self,
        base_loss: nn.Module,
        aux_weights: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.aux_weights = aux_weights or [1.0, 0.4, 0.3, 0.2]

    def forward(
        self,
        logits_list: Union[torch.Tensor, List[torch.Tensor]],
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if not isinstance(logits_list, (list, tuple)):
            result = self.base_loss(logits_list, targets)
            if isinstance(result, dict):
                return result
            return {"total": result}

        total: Optional[torch.Tensor] = None
        first: Optional[Dict] = None
        for i, logits in enumerate(logits_list):
            if i >= len(self.aux_weights):
                break
            w = self.aux_weights[i]
            if logits.shape[-2:] != targets.shape[-2:]:
                logits = F.interpolate(
                    logits, size=targets.shape[-2:], mode="bilinear", align_corners=False
                )
            result = self.base_loss(logits, targets)
            if isinstance(result, dict):
                weighted = w * result["total"]
                if first is None:
                    first = result
            else:
                weighted = w * result
            total = weighted if total is None else total + weighted

        assert total is not None
        if first is not None:
            return {
                "total": total,
                "lovasz": first.get("lovasz", torch.tensor(0.0)),
                "dice": first.get("dice", torch.tensor(0.0)),
                "focal": first.get("focal", torch.tensor(0.0)),
            }
        return {"total": total}


# ============================================================
# Teacher multi-task loss
# ============================================================

class TeacherLossV3(nn.Module):
    """Multi-task loss for the Siamese teacher (Phase 2).

    Combines damage segmentation, binary change detection, and disaster
    type classification losses.

    Args:
        num_damage_classes: Damage severity classes.
        damage_weight: Weight for the damage segmentation sub-loss.
        change_weight: Weight for the binary change detection sub-loss.
        disaster_weight: Weight for the disaster type classification sub-loss.
        focal_gamma: Focal loss gamma for the damage head.
        damage_class_weights: Per-class weights for damage loss.
        use_deep_supervision: Wrap damage loss in deep supervision.
    """

    def __init__(
        self,
        num_damage_classes: int = 6,
        damage_weight: float = 0.70,
        change_weight: float = 0.20,
        disaster_weight: float = 0.10,
        focal_gamma: float = 2.0,
        damage_class_weights: Optional[List[float]] = None,
        use_deep_supervision: bool = True,
    ) -> None:
        super().__init__()
        self.w_damage = damage_weight
        self.w_change = change_weight
        self.w_disaster = disaster_weight

        base = ComboDamageLossV3(
            num_classes=num_damage_classes,
            class_weights=damage_class_weights,
            focal_gamma=focal_gamma,
        )
        self.combo_damage = (
            DeepSupervisionLoss(base) if use_deep_supervision else base
        )
        self.ce_change = nn.CrossEntropyLoss()
        self.ce_disaster = nn.CrossEntropyLoss()

    def forward(
        self,
        outputs: Dict[str, Union[torch.Tensor, List[torch.Tensor]]],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-task teacher loss.

        Args:
            outputs: Model output dict from
                :class:`~afetsonar.models.teacher.SiameseTeacherSegformerV3`.
            targets: Dict with keys ``"damage_mask"`` ``(B, H, W)``,
                ``"change_mask"`` ``(B, H, W)``, ``"disaster_idx"`` ``(B,)``.

        Returns:
            Dict with keys ``total``, ``damage``, ``damage_lovasz``,
            ``damage_dice``, ``damage_focal``, ``change``, ``disaster``.
        """
        dmg = self.combo_damage(outputs["damage_logits"], targets["damage_mask"])
        chg = self.ce_change(outputs["change_logits"], targets["change_mask"])
        dis = self.ce_disaster(outputs["disaster_logits"], targets["disaster_idx"])
        total = self.w_damage * dmg["total"] + self.w_change * chg + self.w_disaster * dis
        return {
            "total": total,
            "damage": dmg["total"].detach(),
            "damage_lovasz": dmg.get("lovasz", torch.tensor(0.0)),
            "damage_dice": dmg.get("dice", torch.tensor(0.0)),
            "damage_focal": dmg.get("focal", torch.tensor(0.0)),
            "change": chg.detach(),
            "disaster": dis.detach(),
        }
