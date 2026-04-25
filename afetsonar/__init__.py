"""AFETSONAR — Drone-based Disaster Damage Assessment & Routing.

Teknofest 2025 · Built in 11 days.

Quick start::

    from afetsonar import AfetsonarPipeline

    pipeline = AfetsonarPipeline("checkpoints/student/student_v1_best_ema.pth")
    map_path = pipeline.generate_map(
        post_path="post_disaster.png",
        pre_path="pre_disaster.png",
        bbox_latlon=(41.003, 28.975, 41.008, 28.981),
        hospitals=[{"name": "Cerrahpaşa", "lat": 41.0048, "lon": 28.9510}],
        output_path="results/disaster_map.html",
    )

Key metrics (xBD test set, 1375 images):
    Teacher  — mIoU_no_bg: 0.424 · mF1: 0.640 · params: 50.3M
    Student  — mIoU_no_bg: 0.395 · mF1: 0.617 · params: 4.3M · latency: 36ms
"""

__version__ = "1.0.0"
__author__ = "AFETSONAR Team"

from afetsonar.pipeline import AfetsonarPipeline
from afetsonar.config import DefaultConfig, NUM_CLASSES, CLASS_NAMES

__all__ = [
    "AfetsonarPipeline",
    "DefaultConfig",
    "NUM_CLASSES",
    "CLASS_NAMES",
    "__version__",
]
