"""AFETSONAR routing engine.

Public API::

    from afetsonar.routing import compute_priority, score_buildings
    from afetsonar.routing import assign_teams, assign_hospitals
    from afetsonar.routing import apply_gradient_weights, astar_segment
    from afetsonar.routing import nearest_neighbor_tsp
    from afetsonar.routing import rank_landing_zones, filter_lz_candidates
"""

from afetsonar.routing.priority import (
    SEVERITY_WEIGHTS,
    DAMAGE_FACTORS,
    compute_priority,
    compute_survival_probability,
    score_buildings,
)
from afetsonar.routing.team_assignment import assign_teams, assign_hospitals
from afetsonar.routing.astar import apply_gradient_weights, astar_segment
from afetsonar.routing.tsp import nearest_neighbor_tsp, tsp_total_distance
from afetsonar.routing.helicopter import (
    filter_lz_candidates,
    rank_landing_zones,
    compute_heli_eta,
)

__all__ = [
    "SEVERITY_WEIGHTS",
    "DAMAGE_FACTORS",
    "compute_priority",
    "compute_survival_probability",
    "score_buildings",
    "assign_teams",
    "assign_hospitals",
    "apply_gradient_weights",
    "astar_segment",
    "nearest_neighbor_tsp",
    "tsp_total_distance",
    "filter_lz_candidates",
    "rank_landing_zones",
    "compute_heli_eta",
]
