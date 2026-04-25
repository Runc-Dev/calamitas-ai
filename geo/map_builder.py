"""8-layer interactive Folium map builder.

This is the class used to generate ``outputs/notebook7/afetsonar_master_map.html``.
The logic in the notebook was largely glue-code; this module packages it
behind a single :class:`FoliumMapBuilder` with 8 dedicated layer helpers:

1. Base tiles (CartoDB + OSM + Satellite)
2. Damage heatmap (priority-scored)
3. Building polygons coloured by damage class
4. Voronoi / team responsibility zones
5. Road graph with gradient edge weights
6. Routed team chains (A* + TSP sequence)
7. Helicopter landing zones (NATO STANAG 3204)
8. Hospitals / shelters

The heavy-lifting algorithms live in :mod:`afetsonar.routing` — this module
only consumes their outputs and renders them to HTML.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union


class FoliumMapBuilder:
    """Construct the AFETSONAR master map one layer at a time.

    Parameters
    ----------
    center:
        ``(latitude, longitude)`` to center the map on.
    zoom_start:
        Initial zoom level (default 15).
    """

    # Colours chosen to remain colour-blind accessible (Viridis-like palette).
    DAMAGE_COLORS = {
        0: "#cccccc",  # background
        1: "#2ecc71",  # no damage
        2: "#f1c40f",  # minor
        3: "#e67e22",  # major
        4: "#e74c3c",  # destroyed
        5: "#95a5a6",  # unclassified
    }

    def __init__(
        self,
        center: Tuple[float, float] = (41.0058, 28.9784),
        zoom_start: int = 15,
    ) -> None:
        try:
            import folium  # noqa: WPS433
        except ImportError as exc:
            raise ImportError("folium paketi gerekli: pip install folium") from exc

        self._folium = folium
        self.map = folium.Map(location=list(center), zoom_start=zoom_start, tiles=None)

        folium.TileLayer("CartoDB positron", name="Light").add_to(self.map)
        folium.TileLayer("OpenStreetMap", name="OSM").add_to(self.map)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}",
            name="Satellite",
            attr="Esri",
        ).add_to(self.map)

    # ------------------------------------------------------------------
    # Layers
    # ------------------------------------------------------------------
    def add_damage_heatmap(
        self,
        points: Iterable[Tuple[float, float, float]],
        name: str = "Damage heatmap",
    ) -> "FoliumMapBuilder":
        """Add a weighted heatmap layer (expects ``(lat, lon, priority)``)."""
        from folium.plugins import HeatMap

        data = [[lat, lon, w] for lat, lon, w in points]
        HeatMap(data, name=name, radius=12, blur=18, min_opacity=0.35).add_to(self.map)
        return self

    def add_building_polygons(
        self,
        buildings: Iterable[Dict[str, Any]],
        name: str = "Buildings",
    ) -> "FoliumMapBuilder":
        """Draw per-building polygons coloured by their predicted damage class.

        Each ``building`` entry must have keys ``polygon`` (list of
        ``(lat, lon)``) and ``damage_class``.
        """
        folium = self._folium
        group = folium.FeatureGroup(name=name)
        for b in buildings:
            color = self.DAMAGE_COLORS.get(int(b.get("damage_class", 0)), "#999999")
            folium.Polygon(
                locations=b["polygon"],
                color=color,
                weight=1,
                fill=True,
                fill_color=color,
                fill_opacity=0.55,
                popup=(
                    f"damage={b.get('damage_class')} "
                    f"priority={b.get('priority', 0):.2f}"
                ),
            ).add_to(group)
        group.add_to(self.map)
        return self

    def add_voronoi(
        self,
        polygons: Iterable[Dict[str, Any]],
        name: str = "Team zones",
    ) -> "FoliumMapBuilder":
        """Draw Voronoi / K-means team responsibility polygons."""
        folium = self._folium
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        group = folium.FeatureGroup(name=name)
        for idx, poly in enumerate(polygons):
            folium.Polygon(
                locations=poly["polygon"],
                color=palette[idx % len(palette)],
                weight=2,
                fill=True,
                fill_opacity=0.1,
                popup=f"Team {poly.get('team_id', idx)}",
            ).add_to(group)
        group.add_to(self.map)
        return self

    def add_road_graph(
        self,
        edges: Iterable[Dict[str, Any]],
        name: str = "Roads",
    ) -> "FoliumMapBuilder":
        """Draw road edges coloured by their gradient / damage weight."""
        folium = self._folium
        group = folium.FeatureGroup(name=name, show=False)
        for edge in edges:
            weight = float(edge.get("weight", 1.0))
            color = "#4682b4" if weight < 1.5 else ("#f39c12" if weight < 3 else "#c0392b")
            folium.PolyLine(
                locations=edge["coords"],
                color=color,
                weight=2 + min(4, weight - 1),
                opacity=0.6,
            ).add_to(group)
        group.add_to(self.map)
        return self

    def add_team_routes(
        self,
        routes: Iterable[Dict[str, Any]],
        name: str = "Team routes",
    ) -> "FoliumMapBuilder":
        """Draw the sequenced A* routes for each team chain."""
        folium = self._folium
        palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        group = folium.FeatureGroup(name=name)
        for idx, route in enumerate(routes):
            color = palette[idx % len(palette)]
            folium.PolyLine(
                locations=route["coords"],
                color=color,
                weight=4,
                opacity=0.85,
                popup=f"Team {route.get('team_id', idx)} · "
                f"{len(route.get('stops', [])):d} stops",
            ).add_to(group)
            for j, stop in enumerate(route.get("stops", [])):
                folium.CircleMarker(
                    location=stop["coord"],
                    radius=5,
                    color=color,
                    fill=True,
                    popup=f"#{j+1} priority={stop.get('priority', 0):.2f}",
                ).add_to(group)
        group.add_to(self.map)
        return self

    def add_landing_zones(
        self,
        zones: Iterable[Dict[str, Any]],
        name: str = "Helicopter LZs",
    ) -> "FoliumMapBuilder":
        """Draw ranked helicopter landing zones (NATO STANAG 3204 criteria)."""
        folium = self._folium
        group = folium.FeatureGroup(name=name)
        for z in zones:
            score = float(z.get("score", 0))
            color = "#27ae60" if score >= 0.8 else ("#f1c40f" if score >= 0.5 else "#c0392b")
            folium.Marker(
                location=z["center"],
                icon=folium.Icon(color="green" if score >= 0.5 else "red", icon="plane"),
                popup=(
                    f"LZ #{z.get('rank', 0)} · area={z.get('area_m2', 0):.0f} m² · "
                    f"score={score:.2f}"
                ),
            ).add_to(group)
            folium.Circle(
                location=z["center"],
                radius=z.get("radius_m", 15),
                color=color,
                fill=True,
                fill_opacity=0.2,
            ).add_to(group)
        group.add_to(self.map)
        return self

    def add_hospitals(
        self,
        hospitals: Iterable[Dict[str, Any]],
        name: str = "Hospitals",
    ) -> "FoliumMapBuilder":
        """Mark hospitals / shelters."""
        folium = self._folium
        group = folium.FeatureGroup(name=name)
        for h in hospitals:
            folium.Marker(
                location=h["coord"],
                icon=folium.Icon(color="blue", icon="plus-sign"),
                popup=h.get("name", "Hospital"),
            ).add_to(group)
        group.add_to(self.map)
        return self

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self, output_path: Optional[Union[str, Path]] = None) -> str:
        """Finalize and (optionally) save the HTML.

        Adds the LayerControl on top of everything so users can toggle the
        8 layers interactively.
        """
        self._folium.LayerControl(collapsed=False).add_to(self.map)
        html = self.map.get_root().render()
        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(html, encoding="utf-8")
        return html


__all__ = ["FoliumMapBuilder"]
