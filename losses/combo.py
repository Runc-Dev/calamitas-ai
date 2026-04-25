"""Combo damage loss + deep-supervision + multi-task teacher loss.

Three classes are exposed here:

- :class:`ComboDamageLossV3` — weighted sum of Lovász + Dice + Focal on the
  damage head. Each term has a complementary failure mode: Lovász optimizes
  mIoU, Dice handles class imbalance in area, and Focal hammers confident
  mistakes on rare classes.
- :class:`DeepSupervisionLoss` — wraps any damage loss so it can be applied
  to a list of logits (main head + aux heads from intermediate decoder
  stages) with geometric decay weights.
- :class:`TeacherLossV3` — assembles the damage combo plus the change and
  disaster auxiliary heads into the single scalar used by Phase 2 training.

References
----------
- Berman et al. 2018 — Lovász-Softmax.
- Milletari et al. 2016 — V-Net / Dice.
- Lin et al. 2017 — Focal Loss.
- Zhao et al. 2017 — PSPNet / auxiliary deep supervision.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from afetsonar.losses.lovasz import LovaszSoftmaxLoss


# ----------------------------------------------------------------------
# Component losses
# ----------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al. 2017)."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Sequence[float]] = None,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        if alpha is not None:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits, targets, reduction="none", ignore_index=self.ignore_index
        )
        p_t = torch.exp(-ce_loss)
        focal_weight = (1.0 - p_t) ** self.gamma
        if self.alpha is not None:
            alpha_t = self.alpha[targets.clamp(min=0)]
            focal_weight = focal_weight * alpha_t
        loss = focal_weight * ce_loss
        valid_mask = (targets != self.ignore_index).float()
        loss = loss * valid_mask
        return loss.sum() / valid_mask.sum().clamp(min=1.0)


class DiceLoss(nn.Module):
    """Macro-average Dice loss, optionally excluding background."""

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = -100,
        smooth: float = 1.0,
        class_weights: Optional[Sequence[float]] = None,
        exclude_background: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.exclude_background = exclude_background
        if class_weights is not None:
            self.register_buffer(
                "class_weights", torch.tensor(class_weights, dtype=torch.float32)
            )
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        valid_mask = targets != self.ignore_index
        targets_clamped = targets.clamp(min=0, max=self.num_classes - 1)
        targets_onehot = F.one_hot(targets_clamped, num_classes=self.num_classes)
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()
        valid_mask_c = valid_mask.unsqueeze(1).float()
        probs = probs * valid_mask_c
        targets_onehot = targets_onehot * valid_mask_c
        dims = (0, 2, 3)
        intersection = (probs * targets_onehot).sum(dims)
        cardinality = probs.sum(dims) + targets_onehot.sum(dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        start_idx = 1 if self.exclude_background else 0
        if self.class_weights is not None:
            weights = self.class_weights[start_idx:]
            dice_loss = (1.0 - dice[start_idx:]) * weights
            return dice_loss.sum() / weights.sum()
        return (1.0 - dice[start_idx:]).mean()


# ----------------------------------------------------------------------
# Combo damage loss v3
# ----------------------------------------------------------------------

class ComboDamageLossV3(nn.Module):
    """Damage head loss: ``lovasz_weight * Lovász + dice_weight * Dice + focal_weight * Focal``.

    All three weights should sum to 1.0 by convention (the defaults do).
    """

    def __init__(
        self,
        num_classes: int,
        class_weights: Optional[Sequence[float]] = None,
        focal_gamma: float = 2.0,
        ignore_index: int = -100,
        lovasz_weight: float = 0.35,
        dice_weight: float = 0.35,
        focal_weight: float = 0.30,
    ) -> None:
        super().__init__()
        self.lovasz = LovaszSoftmaxLoss(
            classes="present",
            ignore_index=ignore_index,
            class_weights=class_weights,
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
        self.lovasz_weight = lovasz_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        lovasz_loss = self.lovasz(logits, targets)
        dice_loss = self.dice(logits, targets)
        focal_loss = self.focal(logits, targets)

        total = (
            self.lovasz_weight * lovasz_loss
            + self.dice_weight * dice_loss
            + self.focal_weight * focal_loss
        )

        return {
            "total": total,
            "lovasz": lovasz_loss.detach(),
            "dice": dice_loss.detach(),
            "focal": focal_loss.detach(),
        }


# ----------------------------------------------------------------------
# Deep-supervision wrapper
# ----------------------------------------------------------------------

class DeepSupervisionLoss(nn.Module):
    """Apply any segmentation loss to a list of prediction heads.

    Parameters
    ----------
    base_loss:
        The inner loss module (e.g. :class:`ComboDamageLossV3`).
    aux_weights:
        Per-head loss weights. The first entry is the main head; subsequent
        entries correspond to auxiliary heads from shallower encoder stages.
        Defaults to ``[1.0, 0.4, 0.3, 0.2]``.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        aux_weights: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.aux_weights = list(aux_weights) if aux_weights else [1.0, 0.4, 0.3, 0.2]

    def forward(
        self,
        logits_list: Union[torch.Tensor, List[torch.Tensor]],
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if not isinstance(logits_list, (list, tuple)):
            return self.base_loss(logits_list, targets)

        total_loss: Optional[torch.Tensor] = None
        first_loss: Optional[Dict[str, torch.Tensor]] = None
        for i, logits in enumerate(logits_list):
            if i >= len(self.aux_weights):
                break
            w = self.aux_weights[i]
            if logits.shape[-2:] != targets.shape[-2:]:
                logits = F.interpolate(
                    logits,
                    size=targets.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            loss_dict = self.base_loss(logits, targets)
            if isinstance(loss_dict, dict):
                weighted = w * loss_dict["total"]
                if first_loss is None:
                    first_loss = loss_dict
            else:
                weighted = w * loss_dict
            total_loss = weighted if total_loss is None else total_loss + weighted

        if first_loss is not None:
            return {
                "total": total_loss,
                "lovasz": first_loss.get("lovasz", torch.tensor(0.0)),
                "dice": first_loss.get("dice", torch.tensor(0.0)),
                "focal": first_loss.get("focal", torch.tensor(0.0)),
            }
        return {"total": total_loss}


# ----------------------------------------------------------------------
# Multi-task teacher loss
# ----------------------------------------------------------------------

class TeacherLossV3(nn.Module):
    """Combined multi-task loss for the SiameseTeacherSegformerV3 model."""

    def __init__(
        self,
        num_damage_classes: int = 6,
        damage_weight: float = 0.70,
        change_weight: float = 0.20,
        disaster_weight: float = 0.10,
        focal_gamma: float = 2.0,
        damage_class_weights: Optional[Sequence[float]] = None,
        use_deep_supervision: bool = False,
    ) -> None:
        super().__init__()
        self.damage_weight = damage_weight
        self.change_weight = change_weight
        self.disaster_weight = disaster_weight
        self.use_deep_supervision = use_deep_supervision

        base_combo = ComboDamageLossV3(
            num_classes=num_damage_classes,
            class_weights=damage_class_weights,
            focal_gamma=focal_gamma,
        )

        if use_deep_supervision:
            self.combo_damage: nn.Module = DeepSupervisionLoss(base_combo)
        else:
            self.combo_damage = base_combo

        self.ce_change = nn.CrossEntropyLoss()
        self.ce_disaster = nn.CrossEntropyLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        damage_logits = outputs["damage_logits"]
        damage_losses = self.combo_damage(damage_logits, targets["damage_mask"])

        change_loss = self.ce_change(outputs["change_logits"], targets["change_mask"])
        disaster_loss = self.ce_disaster(
            outputs["disaster_logits"], targets["disaster_idx"]
        )

        total = (
            self.damage_weight * damage_losses["total"]
            + self.change_weight * change_loss
            + self.disaster_weight * disaster_loss
        )

        return {
            "total": total,
            "damage": damage_losses["total"].detach(),
            "damage_lovasz": damage_losses.get("lovasz", torch.tensor(0.0)),
            "damage_dice": damage_losses.get("dice", torch.tensor(0.0)),
            "damage_focal": damage_losses.get("focal", torch.tensor(0.0)),
            "change": change_loss.detach(),
            "disaster": disaster_loss.detach(),
        }
