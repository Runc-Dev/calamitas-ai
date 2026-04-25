"""AFETSONAR model zoo.

Public API::

    from afetsonar.models import LocalizerSegformer
    from afetsonar.models import SiameseTeacherSegformerV3
    from afetsonar.models import StudentSiameseSegformer
    from afetsonar.models import ModelEMA
"""

from afetsonar.models.ema import ModelEMA
from afetsonar.models.segformer import LocalizerSegformer
from afetsonar.models.student import StudentSiameseSegformer
from afetsonar.models.teacher import SiameseTeacherSegformerV3

__all__ = [
    "LocalizerSegformer",
    "SiameseTeacherSegformerV3",
    "StudentSiameseSegformer",
    "ModelEMA",
]
