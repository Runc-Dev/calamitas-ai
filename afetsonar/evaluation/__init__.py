"""AFETSONAR evaluation utilities.

Public API::

    from afetsonar.evaluation import SegmentationMetrics, ClassificationMetrics
    from afetsonar.evaluation import build_ablation_table, build_sota_table
"""

from afetsonar.evaluation.metrics import ClassificationMetrics, SegmentationMetrics
from afetsonar.evaluation.ablation import (
    ABLATION_HISTORY,
    SOTA_COMPARISON,
    build_ablation_table,
    build_sota_table,
    save_ablation_results,
)

__all__ = [
    "SegmentationMetrics",
    "ClassificationMetrics",
    "ABLATION_HISTORY",
    "SOTA_COMPARISON",
    "build_ablation_table",
    "build_sota_table",
    "save_ablation_results",
]
