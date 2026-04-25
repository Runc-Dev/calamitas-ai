"""Loss functions used across AFETSONAR training stages.

Phase 1 — Localization
    :class:`LocalizationLoss`       CE + Dice on building binary mask.

Phase 2 — Teacher
    :class:`ComboDamageLossV3`      Lovász + Dice + Focal on damage head.
    :class:`DeepSupervisionLoss`    Wrapper for auxiliary head losses.
    :class:`TeacherLossV3`          Multi-task teacher loss assembly.

Phase 3 — Knowledge Distillation
    :class:`KnowledgeDistillationLoss`   5-component KD loss.

Utility
    :func:`derive_change_mask_v2`   Build change mask from damage labels.
    :func:`derive_building_mask`    Build binary building mask.
"""

from afetsonar.losses.combo import (
    ComboDamageLossV3,
    DeepSupervisionLoss,
    DiceLoss,
    FocalLoss,
    TeacherLossV3,
)
from afetsonar.losses.distillation import KnowledgeDistillationLoss
from afetsonar.losses.localization import (
    LocalizationLoss,
    derive_building_mask,
    derive_change_mask_v2,
)
from afetsonar.losses.lovasz import LovaszSoftmaxLoss

__all__ = [
    "ComboDamageLossV3",
    "DeepSupervisionLoss",
    "DiceLoss",
    "FocalLoss",
    "KnowledgeDistillationLoss",
    "LocalizationLoss",
    "LovaszSoftmaxLoss",
    "TeacherLossV3",
    "derive_building_mask",
    "derive_change_mask_v2",
]
