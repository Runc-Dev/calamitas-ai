"""Coordinate geometry, EXIF GPS, drone footprint helpers.

This module deliberately avoids importing heavy GIS libraries at import
time. Functions that need ``exifread``, ``pyproj`` or ``rasterio`` import
them lazily so the rest of the package can be used on machines without a
GIS stack.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ----------------------------------------------------------------------
# EXIF / drone GPS
# ----------------------------------------------------------------------

def _convert_to_degrees(value: Any) -> Optional[float]:
    """Convert EXIF GPS rational (``degrees, minutes, seconds``) to decimal."""
    try:
        d = float(value[0])
        m = float(value[1])
        s = float(value[2])
        return d + (m / 60.0) + (s / 3600.0)
    except (TypeError, IndexError, ValueError):
        return None


def read_exif_gps(image_path: str) -> Optional[Dict[str, float]]:
    """Read GPS coordinates from a drone image's EXIF metadata.

    Returns
    -------
    dict or None
        ``{"latitude": lat, "longitude": lon, "altitude": alt}`` or ``None``
        when no GPS data is present.
    """
    try:
        import exifread  # noqa: WPS433 — lazy import
    except ImportError as exc:
        raise ImportError("exifread paketi gerekli: pip install exifread") from exc

    if not os.path.exists(image_path):
        return None

    try:
        with open(image_path, "rb") as f:
            tags = exifread.process_file(f, details=False, stop_tag="GPS GPSAltitude")
    except Exception:
        return None

    if not tags:
        return None

    lat_tag = tags.get("GPS GPSLatitude")
    lat_ref = tags.get("GPS GPSLatitudeRef")
    if lat_tag is None or lat_ref is None:
        return None

    latitude = _convert_to_degrees(lat_tag.values)
    if latitude is None:
        return None
    if str(lat_ref) == "S":
        latitude = -latitude

    lon_tag = tags.get("GPS GPSLongitude")
    lon_ref = tags.get("GPS GPSLongitudeRef")
    if lon_tag is None or lon_ref is None:
        return None

    longitude = _convert_to_degrees(lon_tag.values)
    if longitude is None:
        return None
    if str(lon_ref) == "W":
        longitude = -longitude

    altitude: Optional[float] = None
    alt_tag = tags.get("GPS GPSAltitude")
    if alt_tag is not None:
        try:
            alt_val = alt_tag.values[0]
            altitude = float(alt_val.num) / float(alt_val.den)
            alt_ref = tags.get("GPS GPSAltitudeRef")
            if alt_ref is not None and int(str(alt_ref)) == 1:
                altitude = -altitude
        except Exception:
            altitude = None

    return {"latitude": latitude, "longitude": longitude, "altitude": altitude}


def read_exif_camera_info(image_path: str) -> Dict[str, Any]:
    """Read camera/drone metadata (model, focal length, image size, timestamp)."""
    try:
        import exifread  # noqa: WPS433
    except ImportError:
        return {}

    if not os.path.exists(image_path):
        return {}

    try:
        with open(image_path, "rb") as f:
            tags = exifread.process_file(f, details=False)
    except Exception:
        return {}

    info: Dict[str, Any] = {}
    if "Image Make" in tags:
        info["make"] = str(tags["Image Make"])
    if "Image Model" in tags:
        info["model"] = str(tags["Image Model"])
    if "EXIF FocalLength" in tags:
        try:
            fl = tags["EXIF FocalLength"].values[0]
            info["focal_length_mm"] = float(fl.num) / float(fl.den)
        except Exception:
            pass
    if "EXIF ExifImageWidth" in tags:
        info["image_width"] = int(str(tags["EXIF ExifImageWidth"]))
    if "EXIF ExifImageLength" in tags:
        info["image_height"] = int(str(tags["EXIF ExifImageLength"]))
    if "EXIF DateTimeOriginal" in tags:
        info["datetime"] = str(tags["EXIF DateTimeOriginal"])

    return info


# ----------------------------------------------------------------------
# Pixel <-> geographic coordinates
# ----------------------------------------------------------------------

def pixel_to_geo(
    pixel_x: float,
    pixel_y: float,
    transform: List[float],
) -> Tuple[float, float]:
    """Apply a 6-element affine transform ``pixel -> geographic``.

    The transform follows the rasterio / GDAL convention:
    ``x_geo = a*px + b*py + c``, ``y_geo = d*px + e*py + f``.
    """
    a, b, c, d, e, f = transform[:6]
    return a * pixel_x + b * pixel_y + c, d * pixel_x + e * pixel_y + f


def geo_to_pixel(
    x_geo: float,
    y_geo: float,
    transform: List[float],
) -> Tuple[float, float]:
    """Inverse of :func:`pixel_to_geo`."""
    a, b, c, d, e, f = transform[:6]
    det = a * e - b * d
    if abs(det) < 1e-12:
        raise ValueError("Singular affine transform, cannot invert")
    pixel_x = (e * (x_geo - c) - b * (y_geo - f)) / det
    pixel_y = (-d * (x_geo - c) + a * (y_geo - f)) / det
    return pixel_x, pixel_y


def pixel_polygon_to_geo(
    pixel_polygon: List[Tuple[float, float]],
    transform: List[float],
) -> List[Tuple[float, float]]:
    """Transform a polygon from pixel space to geographic coordinates."""
    return [pixel_to_geo(px, py, transform) for px, py in pixel_polygon]


# ----------------------------------------------------------------------
# WGS84 <-> UTM
# ----------------------------------------------------------------------

def get_utm_zone(longitude: float, latitude: float) -> Tuple[int, str]:
    """Return the UTM zone number and hemisphere (``'N'``/``'S'``) of a point."""
    zone_number = int((longitude + 180) / 6) + 1
    hemisphere = "N" if latitude >= 0 else "S"
    return zone_number, hemisphere


def wgs84_to_utm(
    latitude: float, longitude: float
) -> Tuple[float, float, int, str]:
    """Convert WGS84 lat/lon to UTM easting/northing (metres)."""
    try:
        from pyproj import Transformer  # noqa: WPS433
    except ImportError as exc:
        raise ImportError("pyproj paketi gerekli: pip install pyproj") from exc

    zone, hemi = get_utm_zone(longitude, latitude)
    epsg = 32600 + zone if hemi == "N" else 32700 + zone
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    easting, northing = transformer.transform(longitude, latitude)
    return easting, northing, zone, hemi


def utm_to_wgs84(
    easting: float,
    northing: float,
    zone_number: int,
    hemisphere: str = "N",
) -> Tuple[float, float]:
    """Inverse of :func:`wgs84_to_utm`."""
    try:
        from pyproj import Transformer  # noqa: WPS433
    except ImportError as exc:
        raise ImportError("pyproj paketi gerekli") from exc

    epsg = 32600 + zone_number if hemisphere == "N" else 32700 + zone_number
    transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    longitude, latitude = transformer.transform(easting, northing)
    return latitude, longitude


def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Great-circle distance between two WGS84 points in metres (Sinnott 1984)."""
    earth_radius_m = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return earth_radius_m * c


# ----------------------------------------------------------------------
# Drone footprint
# ----------------------------------------------------------------------

def estimate_drone_footprint(
    altitude_m: float,
    focal_length_mm: float,
    sensor_width_mm: float,
    image_width_px: int,
    image_height_px: int,
) -> Tuple[float, float, float]:
    """Estimate the ground footprint of a drone image.

    Returns
    -------
    tuple(float, float, float)
        ``(footprint_width_m, footprint_height_m, gsd_cm_per_pixel)`` —
        GSD is the ground sample distance, i.e. how many centimetres each
        pixel covers.
    """
    gsd_cm = (sensor_width_mm * altitude_m * 100) / (focal_length_mm * image_width_px)
    footprint_width_m = (gsd_cm * image_width_px) / 100
    footprint_height_m = (gsd_cm * image_height_px) / 100
    return footprint_width_m, footprint_height_m, gsd_cm


def build_image_index(
    image_dir: str,
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tif", ".tiff"),
) -> List[Dict[str, Any]]:
    """Walk ``image_dir`` and build an index of (path, GPS) pairs.

    The function reads EXIF from JPEGs and GeoTIFF metadata from ``.tif`` /
    ``.tiff`` files. PNGs are returned without GPS.
    """
    image_dir_path = Path(image_dir)
    if not image_dir_path.exists():
        return []

    # Local import to avoid circular dependency with geotiff.py
    from afetsonar.geo.geotiff import read_geotiff_metadata

    records: List[Dict[str, Any]] = []
    for ext in extensions:
        for img_path in image_dir_path.rglob(f"*{ext}"):
            record: Dict[str, Any] = {
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
                if gps is not None:
                    record.update(
                        {
                            "has_gps": True,
                            "latitude": gps["latitude"],
                            "longitude": gps["longitude"],
                            "altitude": gps.get("altitude"),
                            "source": "exif",
                        }
                    )

            elif ext.lower() in (".tif", ".tiff"):
                try:
                    meta = read_geotiff_metadata(str(img_path))
                    if meta.get("crs"):
                        bounds = meta["bounds"]
                        cx = (bounds["left"] + bounds["right"]) / 2
                        cy = (bounds["bottom"] + bounds["top"]) / 2

                        if "4326" not in str(meta["crs"]):
                            try:
                                from pyproj import Transformer  # noqa: WPS433

                                t = Transformer.from_crs(
                                    meta["crs"], "EPSG:4326", always_xy=True
                                )
                                lon, lat = t.transform(cx, cy)
                            except Exception:
                                lat, lon = cy, cx
                        else:
                            lat, lon = cy, cx

                        record.update(
                            {
                                "has_gps": True,
                                "latitude": lat,
                                "longitude": lon,
                                "source": "geotiff",
                            }
                        )
                except Exception:
                    pass

            records.append(record)

    return records


__all__ = [
    "build_image_index",
    "estimate_drone_footprint",
    "geo_to_pixel",
    "get_utm_zone",
    "haversine_distance",
    "pixel_polygon_to_geo",
    "pixel_to_geo",
    "read_exif_camera_info",
    "read_exif_gps",
    "utm_to_wgs84",
    "wgs84_to_utm",
]
