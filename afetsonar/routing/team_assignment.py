"""Rescue team assignment using priority-weighted K-means clustering.

Clusters damaged buildings into N zones, one per rescue team.  Each cluster
centre is the priority-weighted centroid of its buildings.  Teams are then
matched to the nearest hospital for final staging.

References
----------
- MacQueen 1967 — Some methods for classification and analysis of multivariate
  observations (K-means).
- Voronoi 1908 — Nouvelles applications des paramètres continus à la théorie
  des formes quadratiques (Voronoi diagrams).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# Haversine helper (avoid circular import with geo.utils)
# ============================================================

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ============================================================
# K-means clustering
# ============================================================

def assign_teams(
    buildings: List[Dict],
    n_teams: int = 5,
    max_iter: int = 100,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """Priority-weighted K-means team assignment.

    Each building contributes to the cluster centroid proportionally to its
    ``priority_score + 1`` (the ``+1`` prevents zero-weight buildings from
    being completely ignored).

    Args:
        buildings: List of building dicts with keys ``lat``, ``lon``,
            ``priority_score`` (float), ``building_id``.
        n_teams: Number of rescue teams (clusters).
        max_iter: Maximum K-means iterations.
        seed: Random seed for reproducible centroid initialisation.

    Returns:
        ``(buildings, teams)`` where each building dict gains a ``team_id``
        key, and ``teams`` is a list of team dicts with keys ``team_id``,
        ``lat``, ``lon``, ``n_buildings``, ``total_priority``.
    """
    if not buildings:
        return buildings, []

    rng = np.random.default_rng(seed)
    pts = np.array([[b["lat"], b["lon"]] for b in buildings], dtype=np.float64)
    weights = np.array([b.get("priority_score", 0.0) + 1.0 for b in buildings])

    # Initialise centres with weighted random sample (KMeans++ style)
    center_idx = rng.choice(len(buildings), size=min(n_teams, len(buildings)), replace=False)
    centers = pts[center_idx].copy()

    for _ in range(max_iter):
        # Assignment step
        dists = np.array([[_haversine(p[0], p[1], c[0], c[1]) for c in centers] for p in pts])
        labels = dists.argmin(axis=1)

        # Update step (weighted centroid)
        new_centers = centers.copy()
        for k in range(len(centers)):
            mask = labels == k
            if mask.sum() == 0:
                continue
            w = weights[mask]
            new_centers[k] = (pts[mask] * w[:, None]).sum(axis=0) / w.sum()

        if np.allclose(centers, new_centers, atol=1e-8):
            break
        centers = new_centers

    # Write team_id back to buildings
    for i, b in enumerate(buildings):
        b["team_id"] = int(labels[i])

    # Build team summary
    teams = []
    for k in range(len(centers)):
        mask = labels == k
        if mask.sum() == 0:
            continue
        teams.append(
            {
                "team_id": k,
                "lat": float(centers[k, 0]),
                "lon": float(centers[k, 1]),
                "n_buildings": int(mask.sum()),
                "total_priority": float(weights[mask].sum() - mask.sum()),  # subtract the +1 offsets
                "color": _TEAM_COLORS[k % len(_TEAM_COLORS)],
            }
        )

    return buildings, teams


_TEAM_COLORS = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261", "#6a4c93"]


# ============================================================
# Hospital matching
# ============================================================

def assign_hospitals(
    teams: List[Dict],
    hospitals: List[Dict],
) -> List[Dict]:
    """Match each team to its nearest hospital.

    Args:
        teams: Team dicts (output of :func:`assign_teams`).
        hospitals: List of hospital dicts with keys ``lat``, ``lon``, ``name``.

    Returns:
        The same ``teams`` list with ``assigned_hospital`` key added.
    """
    for team in teams:
        if not hospitals:
            team["assigned_hospital"] = "Unknown"
            continue
        nearest = min(
            hospitals,
            key=lambda h: _haversine(team["lat"], team["lon"], h["lat"], h["lon"]),
        )
        team["assigned_hospital"] = nearest["name"]
    return teams
