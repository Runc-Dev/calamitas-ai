"""Geographic utility functions for AFETSONAR.

Covers:
- Haversine great-circle distance (Sinnott 1984).
- Pixel ↔ geographic coordinate conversion via affine transforms.
- WGS84 ↔ UTM projection (pyproj).
- Drone EXIF GPS extraction.
- GSD (ground sample distance) computation.
- Image index builder.

References
----------
- Sinnott 1984 — Virtues of the Haversine.  Sky & Telescope 68(2).
- Pix4D Knowledge Base — GSD calculation methodology.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ============================================================
# Haversine distance
# ============================================================

def haversine_distance(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
) -> float:
    """Great-circle distance between two WGS84 points in **metres**.

    Args:
        lat1: Latitude of point A in decimal degrees.
        lon1: Longitude of point A in decimal degrees.
        lat2: Latitude of point B in decimal degrees.
        lon2: Longitude of point B in decimal degrees.

    Returns:
        Distance in metres.

    References:
        Sinnott 1984 — Virtues of the Haversine.
    """
    R = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ============================================================
# Pixel ↔ geographic coordinate conversion
# ============================================================

def pixel_to_geo(
    pixel_x: float,
    pixel_y: float,
    transform: List[float],
) -> Tuple[float, float]:
    """Convert pixel coordinates to geographic coordinates.

    Uses the standard GDAL/rasterio affine transform:
    ``x_geo = a·px + b·py + c``,  ``y_geo = d·px + e·py + f``.

    Args:
        pixel_x: Column index (0 = left edge).
        pixel_y: Row index (0 = top edge).
        transform: 6-element affine transform ``[a, b, c, d, e, f]``.

    Returns:
        ``(x_geo, y_geo)`` in the CRS of the transform (degrees for WGS84,
        metres for UTM).
    """
    a, b, c, d, e, f = transform[:6]
    return a * pixel_x + b * pixel_y + c, d * pixel_x + e * pixel_y + f


def geo_to_pixel(
    x_geo: float,
    y_geo: float,
    transform: List[float],
) -> Tuple[float, float]:
    """Inverse affine transform: geographic → pixel coordinates.

    Args:
        x_geo: Geographic X (longitude for WGS84).
        y_geo: Geographic Y (latitude for WGS84).
        transform: 6-element affine transform.

    Returns:
        ``(pixel_x, pixel_y)``.

    Raises:
        ValueError: If the transform is singular.
    """
    a, b, c, d, e, f = transform[:6]
    det = a * e - b * d
    if abs(det) < 1e-12:
        raise ValueError("Singular affine transform — cannot invert.")
    px = (e * (x_geo - c) - b * (y_geo - f)) / det
    py = (-d * (x_geo - c) + a * (y_geo - f)) / det
    return px, py


def pixel_polygon_to_geo(
    polygon: List[Tuple[float, float]],
    transform: List[float],
) -> List[Tuple[float, float]]:
    """Convert a pixel-space polygon to geographic coordinates.

    Args:
        polygon: List of ``(x, y)`` pixel coordinate pairs.
        transform: 6-element affine transform.

    Returns:
        List of ``(x_geo, y_geo)`` pairs.
    """
    return [pixel_to_geo(px, py, transform) for px, py in polygon]


# ============================================================
# WGS84 ↔ UTM
# ============================================================

def get_utm_zone(longitude: float, latitude: float) -> Tuple[int, str]:
    """Determine the UTM zone number and hemisphere for a lat/lon point.

    Args:
        longitude: Decimal degrees.
        latitude: Decimal degrees.

    Returns:
        ``(zone_number, hemisphere)`` where hemisphere is ``"N"`` or ``"S"``.
    """
    zone = int((longitude + 180) / 6) + 1
    hemi = "N" if latitude >= 0 else "S"
    return zone, hemi


def wgs84_to_utm(
    latitude: float, longitude: float
) -> Tuple[float, float, int, str]:
    """Project a WGS84 lat/lon point to UTM (metres).

    Args:
        latitude: Decimal degrees.
        longitude: Decimal degrees.

    Returns:
        ``(easting_m, northing_m, zone_number, hemisphere)``.
    """
    try:
        from pyproj import Transformer
    except ImportError:
        raise ImportError("pyproj is required: pip install pyproj")

    zone, hemi = get_utm_zone(longitude, latitude)
    epsg = 32600 + zone if hemi == "N" else 32700 + zone
    t = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    e, n = t.transform(longitude, latitude)
    return e, n, zone, hemi


def utm_to_wgs84(
    easting: float,
    northing: float,
    zone_number: int,
    hemisphere: str = "N",
) -> Tuple[float, float]:
    """Project a UTM point back to WGS84 lat/lon.

    Args:
        easting: Easting in metres.
        northing: Northing in metres.
        zone_number: UTM zone number (1–60).
        hemisphere: ``"N"`` or ``"S"``.

    Returns:
        ``(latitude, longitude)`` in decimal degrees.
    """
    try:
        from pyproj import Transformer
    except ImportError:
        raise ImportError("pyproj is required: pip install pyproj")

    epsg = 32600 + zone_number if hemisphere == "N" else 32700 + zone_number
    t = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    lon, lat = t.transform(easting, northing)
    return lat, lon


# ============================================================
# Drone / GSD utilities
# ============================================================

def compute_gsd(
    altitude_m: float,
    sensor_width_mm: float,
    focal_length_mm: float,
    image_width_px: int,
) -> float:
    """Compute Ground Sample Distance (GSD) in metres per pixel.

    Formula:  ``GSD = (H × s_w) / (f × W_px)``

    Args:
        altitude_m: UAV altitude above ground in metres.
        sensor_width_mm: Camera sensor width in millimetres.
        focal_length_mm: Camera focal length in millimetres.
        image_width_px: Image width in pixels.

    Returns:
        GSD in metres per pixel.

    References:
        Pix4D Knowledge Base — GSD calculation methodology.
    """
    return (altitude_m * sensor_width_mm / 1_000.0) / (focal_length_mm / 1_000.0 * image_width_px)


def read_exif_gps(image_path: str) -> Optional[Dict[str, float]]:
    """Read GPS coordinates from a drone JPEG's EXIF metadata.

    Args:
        image_path: Path to the JPEG/TIFF image.

    Returns:
        Dict with keys ``latitude``, ``longitude``, ``altitude`` (metres),
        or ``None`` if no GPS data is found.
    """
    try:
        import exifread
    except ImportError:
        raise ImportError("exifread is required: pip install exifread")

    if not os.path.exists(image_path):
        return None

    try:
        with open(image_path, "rb") as f:
            tags = exifread.process_file(f, details=False, stop_tag="GPS GPSAltitude")
    except Exception:
        return None

    def _deg(value: list) -> Optional[float]:
        try:
            return float(value[0]) + float(value[1]) / 60.0 + float(value[2]) / 3600.0
        except Exception:
            return None

    lat_tag = tags.get("GPS GPSLatitude")
    lat_ref = tags.get("GPS GPSLatitudeRef")
    lon_tag = tags.get("GPS GPSLongitude")
    lon_ref = tags.get("GPS GPSLongitudeRef")
    if not lat_tag or not lon_tag:
        return None

    lat = _deg(lat_tag.values)
    lon = _deg(lon_tag.values)
    if lat is None or lon is None:
        return None
    if str(lat_ref) == "S":
        lat = -lat
    if str(lon_ref) == "W":
        lon = -lon

    alt: Optional[float] = None
    alt_tag = tags.get("GPS GPSAltitude")
    if alt_tag:
        try:
            v = alt_tag.values[0]
            alt = float(v.num) / float(v.den)
            if tags.get("GPS GPSAltitudeRef") and int(str(tags["GPS GPSAltitudeRef"])) == 1:
                alt = -alt
        except Exception:
            pass

    return {"latitude": lat, "longitude": lon, "altitude": alt}


# ============================================================
# Image index builder
# ============================================================

def build_image_index(
    image_dir: str,
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tif", ".tiff"),
) -> List[Dict]:
    """Walk a directory and build an index of images with GPS coordinates.

    Args:
        image_dir: Root directory to search recursively.
        extensions: File extensions to include.

    Returns:
        List of dicts with keys: ``path``, ``filename``, ``has_gps``,
        ``latitude``, ``longitude``, ``altitude``, ``source``.
    """
    records = []
    for ext in extensions:
        for img_path in Path(image_dir).rglob(f"*{ext}"):
            record: Dict = {
                "path": str(img_path),
                "filename": img_path.name,
                "has_gps": False,
                "latitude": None,
                "longitude": None,
                "altitude": None,
                "source": None,
            }
            if ext.lower() in (".jpg", ".jpeg"):
                gps = read_exif_gps(str(img_path))
                if gps:
                    record.update(
                        has_gps=True,
                        latitude=gps["latitude"],
                        longitude=gps["longitude"],
                        altitude=gps.get("altitude"),
                        source="exif",
                    )
            records.append(record)
    return records
