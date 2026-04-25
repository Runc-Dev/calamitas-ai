"""Building priority scoring (FEMA-based).

The priority score drives team assignment, routing order, and the FEMA
survival probability model.  Higher scores indicate greater urgency.

Priority formula:
    ``priority = severity_weight × area_m² × population_density``

Survival probability (FEMA Survival Curve):
    ``S(t) = S0 × exp(-λ × t) × (1 - damage_factor × building_factor)``

References
----------
- FEMA P-154 (2015) — Rapid Visual Screening of Buildings for Potential
  Seismic Hazards.
- FEMA P-1070 (2016) — USAR Manual.
- AFAD Hızlı Hasar Tespit Kriterleri (HHTK) 2019.
- TÜİK 2023 — İstanbul İl Nüfus Yoğunluğu.
"""

from __future__ import annotations

import math
from typing import Dict, Optional


# ============================================================
# Constants
# ============================================================

#: Severity weights per damage class (FEMA + AFAD calibration).
SEVERITY_WEIGHTS: Dict[int, float] = {
    0: 0.0,   # background
    1: 0.0,   # no_damage
    2: 3.0,   # minor_damage
    3: 7.0,   # major_damage
    4: 10.0,  # destroyed
    5: 2.0,   # unclassified
}

#: Population density (persons/m²) — Fatih / Sultanahmet district (TÜİK 2023).
DEFAULT_POP_DENSITY: float = 0.05

#: FEMA survival curve decay constant (per hour, first 72 h).
FEMA_LAMBDA: float = 0.008

#: Damage factors for the survival curve.
DAMAGE_FACTORS: Dict[int, float] = {
    0: 0.00,  # background
    1: 0.00,  # no_damage
    2: 0.20,  # minor_damage
    3: 0.70,  # major_damage
    4: 0.95,  # destroyed
    5: 0.50,  # unclassified
}

#: Building structure factors (void-space capacity).
BUILDING_K: Dict[str, float] = {
    "concrete": 1.00,
    "masonry": 0.70,
    "wood": 0.50,
    "mixed": 0.85,  # default when structure type is unknown
}


# ============================================================
# Core functions
# ============================================================

def compute_priority(
    damage_class: int,
    area_m2: float,
    pop_density: float = DEFAULT_POP_DENSITY,
) -> float:
    """Compute the priority score for a single building.

    Args:
        damage_class: Integer in 0–5.
        area_m2: Building footprint area in square metres.
        pop_density: Population density in persons/m².

    Returns:
        Dimensionless priority score (higher = more urgent).
    """
    w = SEVERITY_WEIGHTS.get(int(damage_class), 0.0)
    return w * area_m2 * pop_density


def compute_survival_probability(
    damage_class: int,
    elapsed_hours: float = 6.0,
    building_type: str = "mixed",
    lam: float = FEMA_LAMBDA,
    s0: float = 1.0,
) -> float:
    """FEMA survival probability at time ``elapsed_hours`` post-disaster.

    Formula:
        ``S(t) = S0 × exp(-λt) × (1 - damage_factor × building_k)``

    Args:
        damage_class: Integer in 0–5.
        elapsed_hours: Hours since the disaster event.
        building_type: One of ``"concrete"``, ``"masonry"``, ``"wood"``,
            ``"mixed"``.
        lam: Decay constant (default 0.008 per hour).
        s0: Initial survival probability (default 1.0).

    Returns:
        Survival probability in [0, 1].

    References:
        FEMA P-1070 (2016), AFAD HHTK 2019.
    """
    d_factor = DAMAGE_FACTORS.get(int(damage_class), 0.0)
    if d_factor == 0.0:
        return 1.0  # undamaged — full survival

    b_k = BUILDING_K.get(building_type, BUILDING_K["mixed"])
    time_decay = math.exp(-lam * elapsed_hours)
    structural_reduction = 1.0 - d_factor * b_k
    return max(0.0, min(1.0, s0 * time_decay * structural_reduction))


def score_buildings(
    buildings: list,
    elapsed_hours: float = 6.0,
    building_type: str = "mixed",
    pop_density: float = DEFAULT_POP_DENSITY,
) -> list:
    """Compute priority and survival scores for a list of building dicts.

    Args:
        buildings: List of dicts with keys ``damage_class`` (int) and
            ``area_m2`` (float).
        elapsed_hours: Hours since the disaster for survival curve.
        building_type: Structural type assumption.
        pop_density: Population density for priority score.

    Returns:
        The same list with ``priority_score`` and ``survival_prob`` keys
        added to each building dict (in-place modification + return).
    """
    for b in buildings:
        cls = int(b.get("damage_class", 0))
        area = float(b.get("area_m2", 0.0))
        b["priority_score"] = compute_priority(cls, area, pop_density)
        b["survival_prob"] = compute_survival_probability(
            cls, elapsed_hours, building_type
        )
    buildings.sort(key=lambda x: x["priority_score"], reverse=True)
    return buildings
