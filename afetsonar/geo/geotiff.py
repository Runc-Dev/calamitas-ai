"""GeoTIFF read/write utilities for AFETSONAR.

Provides lightweight wrappers around ``rasterio`` for reading satellite
imagery metadata and writing geo-referenced prediction outputs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def read_geotiff_metadata(tiff_path: str) -> Dict[str, Any]:
    """Read coordinate metadata from a GeoTIFF file.

    Args:
        tiff_path: Path to the GeoTIFF.

    Returns:
        Dict with keys:

        - ``crs`` — CRS string (e.g. ``"EPSG:4326"``).
        - ``bounds`` — dict with ``left``, ``bottom``, ``right``, ``top``.
        - ``transform`` — 6-element affine list ``[a, b, c, d, e, f]``.
        - ``width``, ``height``, ``count``, ``dtype``.
        - ``pixel_size_x``, ``pixel_size_y`` — pixel size in CRS units.
    """
    try:
        import rasterio
    except ImportError:
        raise ImportError("rasterio is required: pip install rasterio")

    with rasterio.open(tiff_path) as src:
        bounds = src.bounds
        tf = src.transform
        return {
            "crs": str(src.crs) if src.crs else None,
            "bounds": {
                "left": bounds.left,
                "bottom": bounds.bottom,
                "right": bounds.right,
                "top": bounds.top,
            },
            "transform": list(tf)[:6],
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtype": str(src.dtypes[0]),
            "pixel_size_x": abs(tf[0]),
            "pixel_size_y": abs(tf[4]),
        }


def read_geotiff_array(
    tiff_path: str,
    bands: Optional[List[int]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Read a GeoTIFF as a NumPy array with its metadata.

    Args:
        tiff_path: Path to the GeoTIFF.
        bands: 1-indexed list of bands to read.  ``None`` reads all bands.

    Returns:
        ``(array, metadata)`` where ``array`` has shape ``(H, W)`` for a
        single band or ``(H, W, C)`` for multiple bands.
    """
    try:
        import rasterio
    except ImportError:
        raise ImportError("rasterio is required: pip install rasterio")

    with rasterio.open(tiff_path) as src:
        arr = src.read(bands) if bands else src.read()
    if arr.shape[0] == 1:
        arr = arr[0]
    else:
        arr = np.transpose(arr, (1, 2, 0))
    return arr, read_geotiff_metadata(tiff_path)


def write_prediction_geotiff(
    mask: np.ndarray,
    reference_tiff: str,
    output_path: str,
    nodata: Optional[int] = None,
) -> None:
    """Write a segmentation mask as a single-band GeoTIFF.

    The output inherits the CRS and affine transform from ``reference_tiff``.

    Args:
        mask: 2-D numpy array of shape ``(H, W)`` containing class indices.
        reference_tiff: Source GeoTIFF used to copy geospatial metadata.
        output_path: Destination path for the output GeoTIFF.
        nodata: Optional nodata value to embed in the file metadata.
    """
    try:
        import rasterio
        from rasterio.transform import from_bounds
    except ImportError:
        raise ImportError("rasterio is required: pip install rasterio")

    with rasterio.open(reference_tiff) as src:
        profile = src.profile.copy()

    profile.update(
        count=1,
        dtype=str(mask.dtype),
        compress="lzw",
        nodata=nodata,
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mask[np.newaxis, :, :])
