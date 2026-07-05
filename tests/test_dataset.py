"""Tests for XBDDatasetV2 file-loading robustness (review finding #1).

A missing or corrupt image file must fail loudly — silently substituting
a black image would poison training data without any signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from afetsonar.data.dataset import XBDDatasetV2  # noqa: E402


def _write_sample(tmp_path, *, drop: str = ""):
    """Create a 1-row split CSV; optionally omit one of the files."""
    import cv2

    paths = {}
    for name in ("post", "pre", "mask"):
        p = tmp_path / f"{name}.png"
        if name != drop:
            if name == "mask":
                cv2.imwrite(str(p), np.zeros((64, 64), dtype=np.uint8))
            else:
                cv2.imwrite(
                    str(p),
                    np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
                )
        paths[name] = str(p)

    csv = tmp_path / "split.csv"
    pd.DataFrame([{
        "post_path": paths["post"], "pre_path": paths["pre"],
        "mask_path": paths["mask"], "disaster_idx": 0,
        "filename": "sample.png",
    }]).to_csv(csv, index=False)
    return str(csv)


def _make_ds(csv):
    return XBDDatasetV2(
        csv, mode="teacher", augmentation=None,
        image_size=64, building_aware_crop=False,
    )


def test_loads_when_all_files_exist(tmp_path):
    ds = _make_ds(_write_sample(tmp_path))
    item = ds[0]
    assert item["image"].shape == (6, 64, 64)


@pytest.mark.parametrize("missing", ["post", "pre", "mask"])
def test_missing_file_raises(tmp_path, missing):
    ds = _make_ds(_write_sample(tmp_path, drop=missing))
    with pytest.raises(FileNotFoundError, match="missing or corrupt"):
        ds[0]
