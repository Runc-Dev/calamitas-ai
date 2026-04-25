"""Helicopter landing zone (LZ) selection and ranking.

For buildings that are inaccessible by ground vehicle (blocked road network),
AFETSONAR selects the optimal helicopter landing zone using a weighted score:

    ``score = W_dist × (1 - dist/max_dist) + W_area × (area/max_area)``

LZ candidates are sourced from OSM leisure/park/pitch features that meet the
NATO STANAG 3204 minimum dimension requirement (25 × 25 m for light rotary
wing aircraft).

References
----------
- NATO STANAG 3204 (4th ed.) — Minimum standards for helicopter landing
  zones: 25 m × 25 m clear area, obstacle clearance angle ≥ 10°.
- ICAO Annex 2 — Drone altitude limit 120 m AGL in uncontrolled airspace.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple


#: Minimum LZ dimension (metres) per NATO STANAG 3204.
LZ_MIN_DIM_M: float = 25.0

#: Default scoring weights for distance vs. area trade-off.
W_DIST: float = 0.6
W_AREA: float = 0.4


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def filter_lz_candidates(
    osm_features: list,
    min_dim_m: float = LZ_MIN_DIM_M,
) -> List[Dict]:
    """Filter OSM features to viable helicopter LZ candidates.

    Args:
        osm_features: List of OSM feature dicts (from OSMnx or GeoJSON).
            Each must have ``geometry`` (Shapely geometry) and optionally
            ``name``.
        min_dim_m: Minimum width and height in metres (NATO STANAG 3204).

    Returns:
        List of LZ candidate dicts with keys ``lz_id``, ``name``, ``lat``,
        ``lon``, ``area_m2``.
    """
    # Approximate degrees/metre conversion (~41°N)
    lat_m = 111_320.0
    lon_m = 111_320.0 * math.cos(math.radians(41.005))

    candidates = []
    lz_id = 0

    for feat in osm_features:
        geom = feat.get("geometry") or getattr(feat, "geometry", None)
        if geom is None or getattr(geom, "is_empty", False):
            continue

        bounds = geom.bounds  # (minx, miny, maxx, maxy) in degrees
        width_m = (bounds[2] - bounds[0]) * lon_m
        height_m = (bounds[3] - bounds[1]) * lat_m

        if width_m < min_dim_m or height_m < min_dim_m:
            continue

        centroid = geom.centroid
        area_m2 = width_m * height_m  # approximate
        name = feat.get("name") if isinstance(feat, dict) else getattr(feat, "name", None)

        candidates.append(
            {
                "lz_id": lz_id,
                "name": str(name) if name else f"LZ_{lz_id}",
                "lat": float(centroid.y),
                "lon": float(centroid.x),
                "area_m2": float(area_m2),
            }
        )
        lz_id += 1

    return candidates


def rank_landing_zones(
    building: Dict,
    lz_candidates: List[Dict],
    top_k: int = 3,
    w_dist: float = W_DIST,
    w_area: float = W_AREA,
) -> List[Dict]:
    """Rank LZ candidates for a specific inaccessible building.

    Computes a composite score balancing proximity and landing area size.

    Args:
        building: Building dict with keys ``lat``, ``lon``,
            ``building_id``, ``damage_class_name``.
        lz_candidates: List of LZ candidate dicts from
            :func:`filter_lz_candidates`.
        top_k: Number of top LZs to return.
        w_dist: Weight for distance component (0–1).
        w_area: Weight for area component (0–1).

    Returns:
        List of up to ``top_k`` LZ dicts ranked by composite score, each
        with added keys ``rank``, ``distance_m``, ``score``.
    """
    if not lz_candidates:
        return []

    scored = []
    for lz in lz_candidates:
        d = _haversine(
            building["lat"], building["lon"], lz["lat"], lz["lon"]
        )
        scored.append({**lz, "distance_m": d})

    max_dist = max(lz["distance_m"] for lz in scored)
    max_area = max(lz["area_m2"] for lz in scored)

    for lz in scored:
        dist_score = 1.0 - lz["distance_m"] / max(max_dist, 1.0)
        area_score = lz["area_m2"] / max(max_area, 1.0)
        lz["score"] = w_dist * dist_score + w_area * area_score

    scored.sort(key=lambda x: x["score"], reverse=True)
    for i, lz in enumerate(scored[:top_k]):
        lz["rank"] = i + 1

    return scored[:top_k]


def compute_heli_eta(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    speed_kmh: float = 150.0,
) -> float:
    """Compute helicopter flight time in minutes (straight-line).

    Args:
        origin_lat: Departure latitude.
        origin_lon: Departure longitude.
        dest_lat: Destination latitude.
        dest_lon: Destination longitude.
        speed_kmh: Helicopter cruise speed (default 150 km/h —
            AgustaWestland AW139 cruise).

    Returns:
        Flight time in minutes.
    """
    dist_m = _haversine(origin_lat, origin_lon, dest_lat, dest_lon)
    speed_ms = speed_kmh * 1_000.0 / 3_600.0
    return dist_m / speed_ms / 60.0
