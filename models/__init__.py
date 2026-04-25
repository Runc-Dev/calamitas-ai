"""Model zoo for AFETSONAR.

Exposed classes
---------------
- :class:`LocalizerSegformer`          (Phase 1, binary building localizer)
- :class:`SiameseTeacherSegformerV3`   (Phase 2, damage teacher)
- :class:`StudentSiameseSegformer`     (distilled edge model)
- :class:`LightDecoder`                (re-usable SegFormer MLP decoder)
- :class:`ModelEMA`                    (weight-averaging wrapper)
"""

from afetsonar.models.ema import ModelEMA
from afetsonar.models.segformer import LocalizerSegformer
from afetsonar.models.student import (
    LightDecoder,
    StudentSiameseSegformer,
    NUM_CHANGE_CLASSES,
    NUM_DAMAGE_CLASSES,
    NUM_DISASTER_CLASSES,
)
from afetsonar.models.teacher import SiameseTeacherSegformerV3

__all__ = [
    "LocalizerSegformer",
    "SiameseTeacherSegformerV3",
    "StudentSiameseSegformer",
    "LightDecoder",
    "ModelEMA",
    "NUM_DAMAGE_CLASSES",
    "NUM_CHANGE_CLASSES",
    "NUM_DISASTER_CLASSES",
]
