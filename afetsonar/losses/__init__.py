"""AFETSONAR loss functions.

Public API::

    from afetsonar.losses import LovaszSoftmaxLoss
    from afetsonar.losses import ComboDamageLossV3, TeacherLossV3
    from afetsonar.losses import KnowledgeDistillationLoss
    from afetsonar.losses import LocalizationLoss
    from afetsonar.losses import derive_change_mask, derive_building_mask
"""

from afetsonar.losses.lovasz import LovaszSoftmaxLoss
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
    derive_change_mask,
)

__all__ = [
    "LovaszSoftmaxLoss",
    "FocalLoss",
    "DiceLoss",
    "ComboDamageLossV3",
    "DeepSupervisionLoss",
    "TeacherLossV3",
    "KnowledgeDistillationLoss",
    "LocalizationLoss",
    "derive_change_mask",
    "derive_building_mask",
]
