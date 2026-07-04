"""AFETSONAR global configuration — all hyper-parameters and constants.

All project-wide constants live here so that training scripts, the pipeline,
and notebooks can share a single source of truth.  Override values at runtime
by subclassing ``DefaultConfig`` or by loading a YAML file with
``Config.from_yaml()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# ============================================================
# Damage taxonomy
# ============================================================

NUM_CLASSES: int = 6
"""Total damage severity classes (including background)."""

CLASS_NAMES: List[str] = [
    "background",
    "no_damage",
    "minor_damage",
    "major_damage",
    "destroyed",
    "unclassified",
]

#: Per-class training weights (boost rare damage classes).
CLASS_WEIGHTS: List[float] = [0.05, 1.0, 8.0, 5.0, 7.0, 0.5]

#: Number of disaster event types in xBD.
NUM_DISASTER_CLASSES: int = 5


# ============================================================
# Priority & survival model
# ============================================================

#: Severity weights per class for priority score (FEMA + AFAD).
SEVERITY_WEIGHTS: Dict[int, float] = {
    0: 0.0,   # background
    1: 0.0,   # no_damage
    2: 3.0,   # minor_damage
    3: 7.0,   # major_damage
    4: 10.0,  # destroyed
    5: 2.0,   # unclassified
}

#: Default population density (persons/m²) — override per-region for accurate estimates.
POPULATION_DENSITY: float = 0.05

#: FEMA survival curve decay constant (per hour).
FEMA_LAMBDA: float = 0.008


# ============================================================
# Image / sensor
# ============================================================

#: Native xBD Maxar imagery pixel size.
PIXEL_SIZE_M: float = 0.5

#: Default spatial resolution — matches the *student* (SegFormer-B0)
#: training resolution; also the edge-deployment target.
IMAGE_SIZE: int = 512

#: Teacher (SegFormer-B3) native training/inference resolution.
#: Evaluating the teacher below this costs ~0.09 mF1 — verified on the
#: xBD test_v3 split (2026-07-04): 512px -> mF1 0.551, 768px -> mF1 0.640.
TEACHER_IMAGE_SIZE: int = 768

#: ImageNet normalisation (SegFormer pretrained).
IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]


# ============================================================
# Training schedule
# ============================================================

#: Peak learning rates for the three cosine warm-restart cycles.
LR_PEAKS: List[float] = [1e-4, 1e-4, 7e-5]

#: Warm-up duration at the start of training (epochs).
WARMUP_EPOCHS: int = 3

#: Epoch boundaries for cosine restarts.
RESTART_EPOCHS: List[int] = [25, 50, 75]

#: Total training epochs (teacher Phase 2).
TOTAL_EPOCHS: int = 100

#: EMA decay coefficient.
EMA_DECAY: float = 0.999

#: KD temperature for student distillation.
KD_TEMPERATURE: float = 4.0

#: Training batch sizes (adjust to GPU VRAM).
BATCH_SIZE_TEACHER: int = 4
BATCH_SIZE_STUDENT: int = 8


# ============================================================
# Routing / GIS
# ============================================================

#: UAV flight altitude in metres (ICAO Annex 2 uncontrolled airspace limit).
DRONE_ALTITUDE_M: float = 120.0

#: NATO STANAG 3204 minimum helicopter LZ area (25 × 25 m).
LZ_MIN_AREA_M2: float = 625.0

#: Minimum LZ edge dimension (metres).
LZ_MIN_DIM_M: float = 25.0

#: Ground vehicle speed (AFAD 2019 urban response).
VEHICLE_SPEED_KMH: float = 20.0

#: Walking speed (debris areas).
WALKING_SPEED_KMH: float = 4.0

#: Helicopter cruise speed.
HELI_SPEED_KMH: float = 150.0

#: Impassable buffer radius around destroyed buildings (metres).
ENKAZ_BUFFER_M: float = 30.0

#: Max buildings assigned per rescue team.
MAX_BUILDINGS_PER_TEAM: int = 8

#: Number of rescue teams (K-means clusters).
N_TEAMS: int = 5


# ============================================================
# Dataclass-based config
# ============================================================

@dataclass
class DefaultConfig:
    """Dataclass holding all AFETSONAR hyper-parameters.

    Use this for type-safe access in training scripts::

        from afetsonar.config import DefaultConfig
        cfg = DefaultConfig()
        print(cfg.image_size)   # 768
    """

    # Taxonomy
    num_classes: int = NUM_CLASSES
    class_names: List[str] = field(default_factory=lambda: list(CLASS_NAMES))
    class_weights: List[float] = field(default_factory=lambda: list(CLASS_WEIGHTS))
    num_disaster_classes: int = NUM_DISASTER_CLASSES

    # Sensor
    pixel_size_m: float = PIXEL_SIZE_M
    image_size: int = IMAGE_SIZE

    # Training
    lr_peaks: List[float] = field(default_factory=lambda: list(LR_PEAKS))
    warmup_epochs: int = WARMUP_EPOCHS
    restart_epochs: List[int] = field(default_factory=lambda: list(RESTART_EPOCHS))
    total_epochs: int = TOTAL_EPOCHS
    ema_decay: float = EMA_DECAY
    kd_temperature: float = KD_TEMPERATURE
    batch_size_teacher: int = BATCH_SIZE_TEACHER
    batch_size_student: int = BATCH_SIZE_STUDENT

    # Routing
    drone_altitude_m: float = DRONE_ALTITUDE_M
    lz_min_area_m2: float = LZ_MIN_AREA_M2
    lz_min_dim_m: float = LZ_MIN_DIM_M
    vehicle_speed_kmh: float = VEHICLE_SPEED_KMH
    walking_speed_kmh: float = WALKING_SPEED_KMH
    heli_speed_kmh: float = HELI_SPEED_KMH
    enkaz_buffer_m: float = ENKAZ_BUFFER_M
    max_buildings_per_team: int = MAX_BUILDINGS_PER_TEAM
    n_teams: int = N_TEAMS
    population_density: float = POPULATION_DENSITY

    # Paths (populated at runtime)
    checkpoints_dir: str = "checkpoints"
    data_dir: str = "data"
    outputs_dir: str = "outputs"

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "DefaultConfig":
        """Load configuration from a YAML file.

        Args:
            yaml_path: Path to the YAML config file.

        Returns:
            Populated :class:`DefaultConfig` instance.
        """
        import yaml
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        obj = cls()
        for key, value in (data or {}).items():
            if hasattr(obj, key):
                setattr(obj, key, value)
        return obj
