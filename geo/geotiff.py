"""GeoTIFF read helpers (thin wrappers around :mod:`rasterio`)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def read_geotiff_metadata(tiff_path: str) -> Dict[str, Any]:
    """Read coordinate metadata from a GeoTIFF file.

    Returns
    -------
    dict
        Keys: ``crs`` (CRS string or ``None``), ``bounds``
        (``left`` / ``bottom`` / ``right`` / ``top``), ``transform``
        (6-element affine), ``width``, ``height``, ``count``, ``dtype``,
        ``pixel_size_x``, ``pixel_size_y``.
    """
    try:
        import rasterio  # noqa: WPS433
    except ImportError as exc:
        raise ImportError("rasterio paketi gerekli: pip install rasterio") from exc

    with rasterio.open(tiff_path) as src:
        bounds = src.bounds
        transform = src.transform
        return {
            "crs": str(src.crs) if src.crs else None,
            "bounds": {
                "left": bounds.left,
                "bottom": bounds.bottom,
                "right": bounds.right,
                "top": bounds.top,
            },
            "transform": list(transform)[:6],
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtype": str(src.dtypes[0]),
            "pixel_size_x": abs(transform[0]),
            "pixel_size_y": abs(transform[4]),
        }


def read_geotiff_array(
    tiff_path: str,
    bands: Optional[List[int]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Read a GeoTIFF as a numpy array plus its metadata.

    Parameters
    ----------
    tiff_path:
        Path on disk.
    bands:
        1-indexed list of bands to read; ``None`` reads all bands.

    Returns
    -------
    tuple
        ``(array, metadata)`` — array shape ``[H, W]`` for single-band inputs
        and ``[H, W, C]`` for multi-band.
    """
    try:
        import rasterio  # noqa: WPS433
    except ImportError as exc:
        raise ImportError("rasterio paketi gerekli") from exc

    with rasterio.open(tiff_path) as src:
        if bands is None:
            arr = src.read()
        else:
            arr = src.read(bands)

        if arr.shape[0] == 1:
            arr = arr[0]
        else:
            arr = np.transpose(arr, (1, 2, 0))

        return arr, read_geotiff_metadata(tiff_path)


__all__ = ["read_geotiff_array", "read_geotiff_metadata"]
