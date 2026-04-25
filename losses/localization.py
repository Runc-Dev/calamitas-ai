"""Phase 1 binary localization loss + simple mask derivation helpers.

The localizer (:class:`afetsonar.models.LocalizerSegformer`) is trained with
a 50/50 mix of standard cross entropy and a differentiable binary Dice loss
on the ``building`` class. Background pixels count toward CE but not Dice,
so the loss keeps a direct grip on building mask quality even when the
class distribution is dominated by background.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalizationLoss(nn.Module):
    """CE + Dice loss for binary building localization (Phase 1)."""

    def __init__(self, ignore_index: int = -100) -> None:
        super().__init__()
        self.ignore_index = ignore_index

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        ce_loss = F.cross_entropy(
            logits, targets, reduction="mean", ignore_index=self.ignore_index
        )

        probs = F.softmax(logits, dim=1)
        building_prob = probs[:, 1]
        building_target = (targets == 1).float()

        valid = (targets != self.ignore_index).float()
        building_prob = building_prob * valid
        building_target = building_target * valid

        intersection = (building_prob * building_target).sum()
        union = building_prob.sum() + building_target.sum()
        dice_loss = 1.0 - (2.0 * intersection + 1.0) / (union + 1.0)

        total = 0.5 * ce_loss + 0.5 * dice_loss

        return {
            "total": total,
            "ce": ce_loss.detach(),
            "dice": dice_loss.detach(),
        }


def derive_change_mask_v2(damage_mask: torch.Tensor) -> torch.Tensor:
    """Derive a binary change mask (minor / major / destroyed) from damage labels."""
    return ((damage_mask >= 2) & (damage_mask <= 4)).long()


def derive_building_mask(damage_mask: torch.Tensor) -> torch.Tensor:
    """Derive a binary building mask (any non-background class)."""
    return (damage_mask > 0).long()
