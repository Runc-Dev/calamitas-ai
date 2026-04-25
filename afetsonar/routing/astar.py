"""A* path finding with gradient damage edge weights.

Two-step approach:

1. **Edge weighting** — Each road segment gets a ``gradient_weight`` that
   reflects the expected difficulty of traversal.  Segments intersecting
   destroyed-building buffers are set to ``inf`` (blocked); segments near
   major/minor damage are proportionally slowed.

2. **A* routing** — ``astar_segment`` finds the shortest weighted path
   between two OSM graph nodes using the Haversine distance as an
   admissible heuristic.

References
----------
- Hart, Nilsson & Raphael 1968 — A Formal Basis for the Heuristic
  Determination of Minimum Cost Paths.  IEEE TSSC 4(2):100-107.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import networkx as nx


# ============================================================
# Haversine heuristic
# ============================================================

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _nearest_node(graph: nx.MultiDiGraph, lat: float, lon: float) -> int:
    """Return the OSM node ID closest to (lat, lon) using Haversine distance."""
    try:
        import osmnx as ox
        return ox.distance.nearest_nodes(graph, X=lon, Y=lat)
    except Exception:
        best_node, best_d = None, float("inf")
        for n, data in graph.nodes(data=True):
            d = _haversine(lat, lon, data["y"], data["x"])
            if d < best_d:
                best_d = d
                best_node = n
        assert best_node is not None
        return best_node


# ============================================================
# Gradient edge weighting
# ============================================================

def apply_gradient_weights(
    graph: nx.MultiDiGraph,
    buildings: List[Dict],
    destroyed_buffer_m: float = 30.0,
    damage_buffer_m: float = 50.0,
) -> Tuple[nx.MultiDiGraph, Dict[str, int]]:
    """Assign gradient traversal weights to all road edges.

    For each edge:

    - If the edge intersects a destroyed building's buffer zone → weight = inf.
    - Otherwise → weight = length × (1 + damage_density × 2) where
      damage_density ∈ [0, 1] is based on proximity to major/minor damage.

    The original length is preserved as ``original_length`` for ETA calculation.

    Args:
        graph: OSMnx road graph (modified in-place).
        buildings: List of building dicts with keys ``lat``, ``lon``,
            ``damage_class_name`` (str), ``damage_class`` (int).
        destroyed_buffer_m: Impassable buffer around destroyed buildings (m).
        damage_buffer_m: Slow-down buffer around major/minor buildings (m).

    Returns:
        ``(modified_graph, stats)`` where stats has keys ``n_blocked``,
        ``n_slowed``, ``n_normal``.
    """
    try:
        from shapely.geometry import LineString, Point
        from shapely.ops import unary_union
    except ImportError:
        raise ImportError("shapely is required: pip install shapely")

    # Approximate degrees → metres conversion (Sultanahmet ~41°N)
    LON_M = 111_320.0 * math.cos(math.radians(41.005))
    LAT_M = 111_320.0

    def _m_to_deg(m: float) -> float:
        return m / ((LAT_M + LON_M) / 2.0)

    # Pre-build building point objects
    destroyed_points = []
    damage_points = []
    for b in buildings:
        cls = b.get("damage_class_name", "").lower()
        pt = Point(b["lon"], b["lat"])
        if cls == "destroyed":
            destroyed_points.append(pt)
        elif cls in ("major", "major_damage"):
            damage_points.append({"point": pt, "factor": 2.0, "radius": _m_to_deg(damage_buffer_m)})
        elif cls in ("minor", "minor_damage"):
            damage_points.append({"point": pt, "factor": 0.5, "radius": _m_to_deg(damage_buffer_m)})

    # Destroyed union geometry
    if destroyed_points:
        dest_union = unary_union(
            [p.buffer(_m_to_deg(destroyed_buffer_m)) for p in destroyed_points]
        )
    else:
        dest_union = None

    n_blocked = n_slowed = n_normal = 0

    for u, v, k, data in graph.edges(keys=True, data=True):
        # Store original length
        data["original_length"] = data.get("length", 1.0)

        # Build edge geometry
        u_data, v_data = graph.nodes[u], graph.nodes[v]
        edge_geom = LineString([
            (u_data.get("x", 0.0), u_data.get("y", 0.0)),
            (v_data.get("x", 0.0), v_data.get("y", 0.0)),
        ])

        # Check blocked
        if dest_union is not None and edge_geom.intersects(dest_union):
            data["gradient_weight"] = float("inf")
            n_blocked += 1
            continue

        # Compute damage density
        max_density = 0.0
        for dp in damage_points:
            dist = edge_geom.distance(dp["point"])
            if dist < dp["radius"]:
                density = dp["factor"] * (1.0 - dist / dp["radius"])
                max_density = max(max_density, density)

        if max_density > 0.0:
            data["gradient_weight"] = data["original_length"] * (1.0 + 2.0 * max_density)
            n_slowed += 1
        else:
            data["gradient_weight"] = data["original_length"]
            n_normal += 1

    return graph, {"n_blocked": n_blocked, "n_slowed": n_slowed, "n_normal": n_normal}


# ============================================================
# A* segment routing
# ============================================================

def astar_segment(
    graph: nx.MultiDiGraph,
    src_lat: float,
    src_lon: float,
    dst_lat: float,
    dst_lon: float,
) -> Tuple[bool, List[Tuple[float, float]], float]:
    """Find the shortest weighted path between two geographic points.

    Uses ``gradient_weight`` as the edge cost and Haversine distance as the
    admissible A* heuristic.

    Args:
        graph: Road graph with ``gradient_weight`` edge attributes.
        src_lat: Source latitude.
        src_lon: Source longitude.
        dst_lat: Destination latitude.
        dst_lon: Destination longitude.

    Returns:
        ``(success, path_coords, distance_m)`` where ``path_coords`` is a
        list of ``(lat, lon)`` tuples along the route, and ``distance_m``
        is the physical (not weighted) distance in metres.
        Returns ``(False, [], inf)`` if no path exists.
    """
    src_node = _nearest_node(graph, src_lat, src_lon)
    dst_node = _nearest_node(graph, dst_lat, dst_lon)

    def _heuristic(u: int, v: int) -> float:
        ud, vd = graph.nodes[u], graph.nodes[v]
        return _haversine(ud["y"], ud["x"], vd["y"], vd["x"])

    try:
        path = nx.astar_path(
            graph, src_node, dst_node,
            heuristic=_heuristic,
            weight="gradient_weight",
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound, nx.NetworkXError):
        return False, [], float("inf")

    # Collect coordinates and physical distance
    coords = [(graph.nodes[n]["y"], graph.nodes[n]["x"]) for n in path]
    dist = sum(
        list(graph.get_edge_data(a, b).values())[0].get(
            "original_length",
            list(graph.get_edge_data(a, b).values())[0].get("length", 0.0),
        )
        for a, b in zip(path[:-1], path[1:])
    )
    return True, coords, dist
