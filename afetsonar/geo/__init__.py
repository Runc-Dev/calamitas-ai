"""AFETSONAR geo utilities.

Public API::

    from afetsonar.geo import haversine_distance, pixel_to_geo
    from afetsonar.geo import wgs84_to_utm, compute_gsd
    from afetsonar.geo import read_geotiff_metadata, read_geotiff_array
    from afetsonar.geo import FoliumMapBuilder
"""

from afetsonar.geo.utils import (
    build_image_index,
    compute_gsd,
    geo_to_pixel,
    get_utm_zone,
    haversine_distance,
    pixel_polygon_to_geo,
    pixel_to_geo,
    read_exif_gps,
    utm_to_wgs84,
    wgs84_to_utm,
)
try:
    from afetsonar.geo.geotiff import (
        read_geotiff_array,
        read_geotiff_metadata,
        write_prediction_geotiff,
    )
except ImportError:
    pass  # rasterio not installed — geotiff functions unavailable

try:
    from afetsonar.geo.map_builder import FoliumMapBuilder
except ImportError:
    pass  # folium not installed — FoliumMapBuilder unavailable

__all__ = [
    "haversine_distance",
    "pixel_to_geo",
    "geo_to_pixel",
    "pixel_polygon_to_geo",
    "wgs84_to_utm",
    "utm_to_wgs84",
    "get_utm_zone",
    "compute_gsd",
    "read_exif_gps",
    "build_image_index",
    "read_geotiff_metadata",
    "read_geotiff_array",
    "write_prediction_geotiff",
    "FoliumMapBuilder",
]
