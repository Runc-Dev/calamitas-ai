"""Geographic utilities: coordinate transforms, GeoTIFF IO, Folium map builder."""

from afetsonar.geo.geotiff import read_geotiff_array, read_geotiff_metadata
from afetsonar.geo.map_builder import FoliumMapBuilder
from afetsonar.geo.utils import (
    build_image_index,
    estimate_drone_footprint,
    geo_to_pixel,
    get_utm_zone,
    haversine_distance,
    pixel_polygon_to_geo,
    pixel_to_geo,
    read_exif_camera_info,
    read_exif_gps,
    utm_to_wgs84,
    wgs84_to_utm,
)

__all__ = [
    "FoliumMapBuilder",
    "build_image_index",
    "estimate_drone_footprint",
    "geo_to_pixel",
    "get_utm_zone",
    "haversine_distance",
    "pixel_polygon_to_geo",
    "pixel_to_geo",
    "read_exif_camera_info",
    "read_exif_gps",
    "read_geotiff_array",
    "read_geotiff_metadata",
    "utm_to_wgs84",
    "wgs84_to_utm",
]
