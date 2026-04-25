"""Nearest-neighbour TSP heuristic for multi-building routing.

The nearest-neighbour algorithm is a greedy TSP heuristic: starting from the
team's current position, repeatedly visit the closest unvisited building.
This typically yields tours within 20–25% of the optimum in O(n²) time.

References
----------
- Rosenkrantz, Stearns & Lewis 1977 — An Analysis of Several Heuristics for
  the Traveling Salesman Problem.  SIAM J. Comput. 6(3):563-581.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def nearest_neighbor_tsp(
    start_lat: float,
    start_lon: float,
    buildings: List[Dict],
    lat_key: str = "lat",
    lon_key: str = "lon",
    id_key: str = "building_id",
) -> List[int]:
    """Nearest-neighbour TSP ordering for a team's building list.

    Args:
        start_lat: Team starting latitude (e.g. team cluster centre).
        start_lon: Team starting longitude.
        buildings: List of building dicts — must contain ``lat_key``,
            ``lon_key``, and ``id_key`` fields.
        lat_key: Key for latitude in building dicts.
        lon_key: Key for longitude in building dicts.
        id_key: Key for the building identifier.

    Returns:
        Ordered list of building IDs representing the visiting sequence.

    Example:
        >>> order = nearest_neighbor_tsp(41.005, 28.977, buildings)
        >>> # buildings will be visited in this ID order
    """
    remaining = list(buildings)
    order: List[int] = []
    cur_lat, cur_lon = start_lat, start_lon

    while remaining:
        best_idx, best_d = 0, float("inf")
        for i, b in enumerate(remaining):
            d = _haversine(cur_lat, cur_lon, b[lat_key], b[lon_key])
            if d < best_d:
                best_d, best_idx = d, i
        chosen = remaining.pop(best_idx)
        order.append(chosen[id_key])
        cur_lat, cur_lon = chosen[lat_key], chosen[lon_key]

    return order


def tsp_total_distance(
    start_lat: float,
    start_lon: float,
    buildings: List[Dict],
    order: List[int],
    lat_key: str = "lat",
    lon_key: str = "lon",
    id_key: str = "building_id",
) -> float:
    """Compute the approximate ground distance of a TSP tour in metres.

    Uses straight-line Haversine distances (not routed distances) for
    quick comparison between tour orderings.

    Args:
        start_lat: Starting latitude.
        start_lon: Starting longitude.
        buildings: List of building dicts.
        order: Visiting order (list of building IDs).
        lat_key: Latitude key.
        lon_key: Longitude key.
        id_key: Building ID key.

    Returns:
        Total approximate distance in metres.
    """
    b_by_id = {b[id_key]: b for b in buildings}
    cur_lat, cur_lon = start_lat, start_lon
    total = 0.0
    for bid in order:
        b = b_by_id.get(bid)
        if b is None:
            continue
        total += _haversine(cur_lat, cur_lon, b[lat_key], b[lon_key])
        cur_lat, cur_lon = b[lat_key], b[lon_key]
    return total
