"""FoliumMapBuilder — 8-layer interactive disaster response map.

Builds the AFETSONAR master map with the following layers:

1. Satellite basemap (Esri World Imagery).
2. Damage segmentation overlay (6-class colour-coded).
3. Priority scores (building markers, size ∝ score).
4. Rescue team routing (colour-coded A* paths per team).
5. Voronoi team zones.
6. Hospital / assembly point markers.
7. Helicopter landing zones (NATO STANAG 3204 compliant).
8. Alternative routes (k-shortest paths for destroyed/major buildings).

References
----------
- NATO STANAG 3204 — Helicopter landing zone minimum dimensions.
- FEMA P-154 — Rapid Visual Screening for potential hazards.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import folium
from folium import plugins as fp


# Damage class colour palette
DAMAGE_COLORS: Dict[str, str] = {
    "no_damage":    "#4caf50",  # green
    "minor_damage": "#ffeb3b",  # yellow
    "major_damage": "#ff9800",  # orange
    "destroyed":    "#f44336",  # red
    "unclassified": "#9c27b0",  # purple
    "background":   "#9e9e9e",  # grey
}

# Index-aligned with config.CLASS_NAMES (0=background … 5=unclassified)
_CLASS_INDEX_TO_NAME = [
    "background", "no_damage", "minor_damage",
    "major_damage", "destroyed", "unclassified",
]

TEAM_COLORS = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261", "#6a4c93"]


def _damage_name(building: Dict[str, Any]) -> str:
    """Resolve a building's damage class name.

    Pipeline buildings carry ``damage_class`` as an *int* (0–5) plus a
    ``damage_class_name`` string; older callers passed the name directly
    in ``damage_class``.  Accept all three forms.
    """
    name = building.get("damage_class_name")
    if isinstance(name, str) and name:
        return name
    cls = building.get("damage_class", "background")
    if isinstance(cls, (int, float)):
        idx = int(cls)
        if 0 <= idx < len(_CLASS_INDEX_TO_NAME):
            return _CLASS_INDEX_TO_NAME[idx]
        return "background"
    return str(cls)


class FoliumMapBuilder:
    """Build and export an 8-layer interactive Folium map.

    Args:
        center_lat: Map centre latitude.
        center_lon: Map centre longitude.
        zoom_start: Initial zoom level.

    Example:
        >>> builder = FoliumMapBuilder(41.005, 28.977)
        >>> builder.add_damage_markers(buildings_df)
        >>> builder.add_team_routes(routes)
        >>> builder.add_hospitals(hospitals)
        >>> builder.save("afetsonar_master_map.html")
    """

    def __init__(
        self,
        center_lat: float,
        center_lon: float,
        zoom_start: int = 15,
    ) -> None:
        self.map = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom_start,
            tiles=None,
        )
        self._add_basemaps()
        self._layers: Dict[str, folium.FeatureGroup] = {}

    # ------------------------------------------------------------------
    # Basemap setup
    # ------------------------------------------------------------------

    def _add_basemaps(self) -> None:
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
                  "World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery",
            name="Satellite",
        ).add_to(self.map)
        folium.TileLayer("OpenStreetMap", name="Street Map").add_to(self.map)

    def _get_layer(self, name: str) -> folium.FeatureGroup:
        if name not in self._layers:
            self._layers[name] = folium.FeatureGroup(name=name, show=True)
            self._layers[name].add_to(self.map)
        return self._layers[name]

    # ------------------------------------------------------------------
    # Layer adders
    # ------------------------------------------------------------------

    def add_damage_markers(
        self,
        buildings: List[Dict[str, Any]],
        layer_name: str = "🏗️ Damage Assessment",
    ) -> "FoliumMapBuilder":
        """Add building damage markers.

        Args:
            buildings: List of dicts with keys ``lat``, ``lon``,
                ``damage_class``, ``priority_score``, ``area_m2``,
                ``building_id``.
            layer_name: Folium layer name shown in the layer control.

        Returns:
            ``self`` for chaining.
        """
        layer = self._get_layer(layer_name)
        for b in buildings:
            if "lat" not in b or "lon" not in b:
                continue  # not geo-referenced (no bbox was provided)
            name = _damage_name(b)
            color = DAMAGE_COLORS.get(name, "#9e9e9e")
            radius = max(4, min(14, max(b.get("priority_score", 1), 0) ** 0.5 * 2))
            team = b.get("team_id")
            team_txt = f" | Team {team}" if team is not None else ""
            folium.CircleMarker(
                location=[b["lat"], b["lon"]],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.85,
                tooltip=(
                    f"Building #{b.get('building_id', '?')} | {name} | "
                    f"Priority: {b.get('priority_score', 0):.1f} | "
                    f"Area: {b.get('area_m2', 0):.0f} m²{team_txt}"
                ),
            ).add_to(layer)
        return self

    def add_building_footprints(
        self,
        buildings: List[Dict[str, Any]],
        layer_name: str = "🏠 Building Footprints",
    ) -> "FoliumMapBuilder":
        """Draw building boundary polygons colour-coded by damage class.

        Args:
            buildings: Building dicts carrying ``polygon_latlon``
                (``[lat, lon]`` vertex list from ``mask_to_buildings``).
                Buildings without a polygon are skipped — use
                :meth:`add_damage_markers` for centroid fallback.
            layer_name: Folium layer name.

        Returns:
            ``self`` for chaining.
        """
        layer = self._get_layer(layer_name)
        for b in buildings:
            polygon = b.get("polygon_latlon")
            if not polygon or len(polygon) < 3:
                continue
            name = _damage_name(b)
            color = DAMAGE_COLORS.get(name, "#9e9e9e")
            folium.Polygon(
                locations=polygon,
                color=color,
                weight=2,
                fill=True,
                fill_color=color,
                fill_opacity=0.35,
                tooltip=(
                    f"Building #{b.get('building_id', '?')} | {name} | "
                    f"Area: {b.get('area_m2', 0):.0f} m²"
                ),
            ).add_to(layer)
        return self

    def add_team_zones(
        self,
        teams: List[Dict[str, Any]],
        layer_name: str = "👥 Team Zones",
    ) -> "FoliumMapBuilder":
        """Add rescue-team centre markers with assignment summaries.

        Args:
            teams: Team dicts from ``assign_teams`` / ``assign_hospitals``
                with keys ``team_id``, ``lat``, ``lon``, ``n_buildings``,
                ``total_priority`` and optionally ``assigned_hospital``.
            layer_name: Folium layer name.

        Returns:
            ``self`` for chaining.
        """
        layer = self._get_layer(layer_name)
        for t in teams:
            if "lat" not in t or "lon" not in t:
                continue
            tid = t.get("team_id", 0)
            color = t.get("color", TEAM_COLORS[tid % len(TEAM_COLORS)])
            hospital = t.get("assigned_hospital", "?")
            folium.Marker(
                location=[t["lat"], t["lon"]],
                icon=folium.Icon(color="blue", icon="user", prefix="fa"),
                tooltip=(
                    f"Team {tid} | {t.get('n_buildings', 0)} buildings | "
                    f"priority {t.get('total_priority', 0):.0f} | base: {hospital}"
                ),
            ).add_to(layer)
            folium.Circle(
                location=[t["lat"], t["lon"]],
                radius=60,
                color=color,
                fill=True,
                fill_opacity=0.10,
                weight=2,
            ).add_to(layer)
        return self

    def add_team_routes(
        self,
        routes: List[Dict[str, Any]],
        layer_name: str = "🚗 Rescue Routes",
    ) -> "FoliumMapBuilder":
        """Add team routing polylines.

        Args:
            routes: List of dicts with keys ``coords`` (list of [lat, lon]),
                ``team_id``, ``total_m``, ``hospital``.
            layer_name: Folium layer name.

        Returns:
            ``self`` for chaining.
        """
        layer = self._get_layer(layer_name)
        for route in routes:
            coords = route.get("coords", [])
            if len(coords) < 2:
                continue
            tid = route.get("team_id", 0)
            color = TEAM_COLORS[tid % len(TEAM_COLORS)]
            folium.PolyLine(
                locations=coords,
                color=color,
                weight=4,
                opacity=0.85,
                tooltip=(
                    f"Team {tid} | {route.get('total_m', 0):.0f} m → "
                    f"{route.get('hospital', '?')[:30]}"
                ),
            ).add_to(layer)
        return self

    def add_hospitals(
        self,
        hospitals: List[Dict[str, Any]],
        layer_name: str = "🏥 Hospitals & Assembly",
    ) -> "FoliumMapBuilder":
        """Add hospital / assembly point markers.

        Args:
            hospitals: List of dicts with keys ``lat``, ``lon``, ``name``.
            layer_name: Folium layer name.

        Returns:
            ``self`` for chaining.
        """
        layer = self._get_layer(layer_name)
        for h in hospitals:
            folium.Marker(
                location=[h["lat"], h["lon"]],
                icon=folium.Icon(color="green", icon="plus-sign"),
                tooltip=h.get("name", "Hospital"),
            ).add_to(layer)
        return self

    def add_landing_zones(
        self,
        lz_list: List[Dict[str, Any]],
        layer_name: str = "🚁 Helicopter LZ",
    ) -> "FoliumMapBuilder":
        """Add NATO STANAG 3204 compliant landing zone markers.

        Args:
            lz_list: List of dicts with keys ``lat``, ``lon``, ``name``,
                ``area_m2``.
            layer_name: Folium layer name.

        Returns:
            ``self`` for chaining.
        """
        layer = self._get_layer(layer_name)
        for lz in lz_list:
            folium.Marker(
                location=[lz["lat"], lz["lon"]],
                icon=folium.Icon(color="red", icon="plane", prefix="fa"),
                tooltip=(
                    f"LZ: {lz.get('name', '?')} | "
                    f"{lz.get('area_m2', 0):.0f} m²"
                ),
            ).add_to(layer)
        return self

    def add_alternative_routes(
        self,
        alt_routes: List[Dict[str, Any]],
        layer_name: str = "🔀 Alternative Routes",
    ) -> "FoliumMapBuilder":
        """Add k-shortest alternative routes for high-priority buildings.

        Args:
            alt_routes: List of dicts with keys ``coords``, ``building_id``,
                ``rank``, ``physical_m``.
            layer_name: Folium layer name.

        Returns:
            ``self`` for chaining.
        """
        layer = self._get_layer(layer_name)
        rank_styles = {1: ("#ff5722", 3, 0.9), 2: ("#ff9800", 2, 0.6), 3: ("#ffc107", 1.5, 0.4)}
        for ar in alt_routes:
            coords = ar.get("coords", [])
            if len(coords) < 2:
                continue
            rank = ar.get("rank", 1)
            color, weight, opacity = rank_styles.get(rank, ("#bdbdbd", 1.5, 0.4))
            folium.PolyLine(
                locations=coords,
                color=color,
                weight=weight,
                opacity=opacity,
                dash_array="5 5" if rank > 1 else None,
                tooltip=(
                    f"Building #{ar.get('building_id', '?')} "
                    f"Alt #{rank} | {ar.get('physical_m', 0):.0f} m"
                ),
            ).add_to(layer)
        return self

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def add_layer_control(self) -> "FoliumMapBuilder":
        """Add a layer control widget to the map."""
        folium.LayerControl(collapsed=False).add_to(self.map)
        return self

    def save(self, output_path: str) -> str:
        """Save the map as a self-contained HTML file.

        Args:
            output_path: Destination HTML file path.

        Returns:
            The absolute path of the saved file.
        """
        import os
        self.add_layer_control()
        self.map.save(output_path)
        size_kb = os.path.getsize(output_path) / 1024
        print(f"Map saved -> {output_path}  ({size_kb:.1f} KB)")
        return os.path.abspath(output_path)
