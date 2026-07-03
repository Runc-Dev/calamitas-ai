"""xBD dataset preprocessing utilities.

Functions for:

- Building polygon JSON → segmentation mask conversion.
- Per-sample weight computation for ``WeightedRandomSampler``.
- Train/val/test split generation from xBD directory structure.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


# ============================================================
# xBD label → damage class mapping
# ============================================================

XBD_DAMAGE_CLASSES: Dict[str, int] = {
    "no-damage":      1,
    "minor-damage":   2,
    "major-damage":   3,
    "destroyed":      4,
    "un-classified":  5,
}
"""Map xBD JSON label strings to integer class indices (background = 0)."""


def mask_from_json(
    json_path: str,
    image_size: Tuple[int, int] = (1024, 1024),
) -> np.ndarray:
    """Convert an xBD annotation JSON file to a damage segmentation mask.

    The xBD JSON format stores building polygons under the key
    ``"features" → "xy"`` with each feature having a ``"subtype"`` property
    indicating the damage class.

    Args:
        json_path: Path to the xBD annotation JSON (post-disaster label).
        image_size: ``(height, width)`` of the output mask.

    Returns:
        8-bit grayscale numpy array of shape ``image_size`` with pixel values
        in ``{0, 1, 2, 3, 4, 5}``.
    """
    mask = np.zeros(image_size, dtype=np.uint8)  # background = 0

    if not os.path.exists(json_path):
        return mask

    with open(json_path) as f:
        data = json.load(f)

    features = data.get("features", {}).get("xy", [])
    for feat in features:
        props = feat.get("properties", {})
        subtype = props.get("subtype", "no-damage").lower()
        damage_class = XBD_DAMAGE_CLASSES.get(subtype, 1)

        wkt = feat.get("wkt", "")
        polygon = _wkt_to_pixel_polygon(wkt, image_size)
        if polygon is not None and len(polygon) >= 3:
            cv2.fillPoly(mask, [polygon], damage_class)

    return mask


def _wkt_to_pixel_polygon(
    wkt: str,
    image_size: Tuple[int, int],
) -> Optional[np.ndarray]:
    """Parse a WKT POLYGON string into a pixel coordinate array.

    Args:
        wkt: Well-Known Text polygon string.
        image_size: Target ``(height, width)`` — used to clamp coordinates.

    Returns:
        Integer numpy array of shape ``(N, 1, 2)`` suitable for OpenCV
        polygon drawing, or ``None`` if parsing fails.
    """
    try:
        # Extract coordinate block from POLYGON ((x0 y0, x1 y1, ...))
        coords_str = wkt.strip()
        if not coords_str.startswith("POLYGON"):
            return None
        start = coords_str.index("((") + 2
        end = coords_str.rindex("))")
        pairs = coords_str[start:end].strip().split(",")
        pts = []
        h, w = image_size
        for pair in pairs:
            xy = pair.strip().split()
            if len(xy) >= 2:
                x = max(0, min(w - 1, int(float(xy[0]))))
                y = max(0, min(h - 1, int(float(xy[1]))))
                pts.append([x, y])
        if len(pts) < 3:
            return None
        return np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    except Exception:
        return None


# ============================================================
# Sample weight computation
# ============================================================

def compute_sample_weights(
    df: pd.DataFrame,
    severity_weights: Optional[Dict[int, float]] = None,
) -> np.ndarray:
    """Compute per-sample training weights for ``WeightedRandomSampler``.

    Samples containing heavily-damaged buildings receive higher weights so
    rare classes (destroyed, major) are seen proportionally more often.

    Args:
        df: DataFrame with columns ``"max_damage_class"`` (int 0–5) and
            optionally ``"damage_area_fraction"`` (float 0–1).
        severity_weights: Mapping from damage class index to sampling weight.
            Defaults to ``{0:0.1, 1:0.5, 2:2.0, 3:4.0, 4:6.0, 5:1.0}``.

    Returns:
        Float32 numpy array of length ``len(df)``.
    """
    if severity_weights is None:
        severity_weights = {0: 0.1, 1: 0.5, 2: 2.0, 3: 4.0, 4: 6.0, 5: 1.0}

    weights = np.ones(len(df), dtype=np.float32)

    if "max_damage_class" in df.columns:
        base = df["max_damage_class"].map(severity_weights).fillna(1.0).values
        weights = weights * base.astype(np.float32)

    if "damage_area_fraction" in df.columns:
        frac = df["damage_area_fraction"].fillna(0.0).values.astype(np.float32)
        # Multiply by 1 + frac so dense damage scenes get higher weight
        weights = weights * (1.0 + frac)

    return weights


# ============================================================
# Split generation
# ============================================================

def build_split_csv(
    xbd_root: str,
    output_dir: str,
    train_fraction: float = 0.80,
    val_fraction: float = 0.10,
    seed: int = 42,
) -> Tuple[str, str, str]:
    """Walk an xBD directory and produce train/val/test CSV split files.

    Expects the standard xBD directory layout::

        xbd_root/
            train/
                images/
                    <disaster>_<id>_pre_disaster.png
                    <disaster>_<id>_post_disaster.png
                labels/
                    <disaster>_<id>_post_disaster.json   ← damage labels
                targets/
                    <disaster>_<id>_post_disaster.png    ← pre-rendered masks

    Args:
        xbd_root: Path to the xBD root directory.
        output_dir: Directory where CSV files will be written.
        train_fraction: Fraction of data used for training.
        val_fraction: Fraction of data used for validation.
        seed: Random seed for reproducible splits.

    Returns:
        ``(train_csv, val_csv, test_csv)`` paths.
    """
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)

    xbd_root = Path(xbd_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict] = []
    disaster_map: Dict[str, int] = {}

    for split_dir in ("train", "tier3", "hold"):
        images_dir = xbd_root / split_dir / "images"
        targets_dir = xbd_root / split_dir / "targets"
        if not images_dir.exists():
            continue

        post_files = sorted(images_dir.glob("*_post_disaster.png"))
        for post_path in post_files:
            stem = post_path.stem.replace("_post_disaster", "")
            pre_path = images_dir / f"{stem}_pre_disaster.png"
            mask_path = targets_dir / f"{stem}_post_disaster.png"
            if not pre_path.exists() or not mask_path.exists():
                continue

            disaster = stem.rsplit("_", 1)[0] if "_" in stem else stem
            if disaster not in disaster_map:
                disaster_map[disaster] = len(disaster_map)
            d_idx = disaster_map[disaster]

            records.append(
                {
                    "filename": post_path.name,
                    "pre_path": str(pre_path),
                    "post_path": str(post_path),
                    "mask_path": str(mask_path),
                    "disaster": disaster,
                    "disaster_idx": d_idx,
                }
            )

    if not records:
        raise FileNotFoundError(
            f"No xBD images found under {xbd_root}. "
            "Check that 'images/' and 'targets/' sub-directories exist."
        )

    _random.shuffle(records)
    n = len(records)
    n_train = int(n * train_fraction)
    n_val = int(n * val_fraction)

    train_df = pd.DataFrame(records[:n_train])
    val_df = pd.DataFrame(records[n_train:n_train + n_val])
    test_df = pd.DataFrame(records[n_train + n_val:])

    train_csv = str(output_dir / "train.csv")
    val_csv = str(output_dir / "val.csv")
    test_csv = str(output_dir / "test.csv")

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(
        f"Split written → train: {len(train_df)}, val: {len(val_df)}, "
        f"test: {len(test_df)} images"
    )
    return train_csv, val_csv, test_csv
