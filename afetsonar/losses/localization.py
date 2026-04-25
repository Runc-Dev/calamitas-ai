"""Phase 1 building localization loss (binary CE + Dice).

Also provides two utility functions used during Phase 2 preprocessing:

- ``derive_change_mask`` — converts a 6-class damage mask to a binary
  change indicator (0 = undamaged/background, 1 = any structural damage).
- ``derive_building_mask`` — converts a 6-class damage mask to a binary
  building/background indicator (used for Phase 1 supervision).
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalizationLoss(nn.Module):
    """Binary building segmentation loss (BCE + Dice, 50-50 split).

    Args:
        ignore_index: Label value excluded from loss computation.
    """

    def __init__(self, ignore_index: int = -100) -> None:
        super().__init__()
        self.ignore_index = ignore_index

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute localization loss.

        Args:
            logits: ``(B, 2, H, W)`` raw logits (class 0 = bg, 1 = building).
            targets: ``(B, H, W)`` binary labels.

        Returns:
            Dict with keys ``total``, ``ce``, ``dice``.
        """
        l_ce = F.cross_entropy(
            logits, targets, reduction="mean", ignore_index=self.ignore_index
        )

        probs = F.softmax(logits, dim=1)
        bld_prob = probs[:, 1]
        bld_tgt = (targets == 1).float()
        valid = (targets != self.ignore_index).float()
        bld_prob = bld_prob * valid
        bld_tgt = bld_tgt * valid

        inter = (bld_prob * bld_tgt).sum()
        union = bld_prob.sum() + bld_tgt.sum()
        l_dice = 1.0 - (2.0 * inter + 1.0) / (union + 1.0)

        total = 0.5 * l_ce + 0.5 * l_dice
        return {
            "total": total,
            "ce": l_ce.detach(),
            "dice": l_dice.detach(),
        }


# ============================================================
# Mask derivation helpers
# ============================================================

def derive_change_mask(damage_mask: torch.Tensor) -> torch.Tensor:
    """Convert 6-class damage mask to binary change mask.

    Classes 2 (minor), 3 (major), and 4 (destroyed) are considered "changed".
    Background (0), no-damage (1), and unclassified (5) map to 0.

    Args:
        damage_mask: Integer tensor of shape ``(B, H, W)`` with values 0–5.

    Returns:
        Binary tensor of shape ``(B, H, W)`` with values 0 or 1.
    """
    return ((damage_mask >= 2) & (damage_mask <= 4)).long()


def derive_building_mask(damage_mask: torch.Tensor) -> torch.Tensor:
    """Convert 6-class damage mask to binary building/background mask.

    Any pixel with a non-background label (≥ 1) is considered a building.

    Args:
        damage_mask: Integer tensor of shape ``(B, H, W)`` with values 0–5.

    Returns:
        Binary tensor of shape ``(B, H, W)`` with values 0 or 1.
    """
    return (damage_mask > 0).long()
