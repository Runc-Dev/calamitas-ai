"""Integration tests for AfetsonarPipeline.

These tests run on CPU without real checkpoints by creating a temporary
model file containing a minimal StudentSiameseSegformer state dict.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
torch = pytest.importorskip("torch", reason="torch not installed — skipping pipeline tests")


@pytest.fixture
def student_ckpt(tmp_path):
    """Create a temporary checkpoint file with random weights."""
    from afetsonar.models import StudentSiameseSegformer
    model = StudentSiameseSegformer(pretrained=False)
    ckpt_path = str(tmp_path / "test_student.pth")
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
    return ckpt_path


@pytest.fixture
def post_image(tmp_path):
    """Create a small test PNG image."""
    import cv2
    import numpy as np
    img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    path = str(tmp_path / "post.png")
    cv2.imwrite(path, img)
    return path


class TestAfetsonarPipeline:
    def test_predict_returns_mask(self, student_ckpt, post_image):
        from afetsonar import AfetsonarPipeline
        pipeline = AfetsonarPipeline(student_ckpt, device="cpu")
        mask = pipeline.predict(post_image)
        assert isinstance(mask, np.ndarray)
        assert mask.ndim == 2
        assert mask.min() >= 0 and mask.max() <= 5

    def test_mask_to_buildings(self, student_ckpt, post_image):
        from afetsonar import AfetsonarPipeline
        pipeline = AfetsonarPipeline(student_ckpt, device="cpu")
        mask = pipeline.predict(post_image)
        buildings = pipeline.mask_to_buildings(mask)
        assert isinstance(buildings, list)

    def test_mask_to_buildings_with_bbox(self, student_ckpt, post_image):
        from afetsonar import AfetsonarPipeline
        pipeline = AfetsonarPipeline(student_ckpt, device="cpu")
        mask = pipeline.predict(post_image)
        bbox = (41.003, 28.975, 41.008, 28.981)
        buildings = pipeline.mask_to_buildings(mask, bbox_latlon=bbox)
        # Any detected building should have geo coordinates
        for b in buildings:
            assert "lat" in b and "lon" in b

    def test_synthetic_square_polygon_and_area(self, student_ckpt):
        """A hand-drawn square mask must yield a correct footprint polygon
        and a bbox-derived (not fixed 0.5 m/px) area."""
        from afetsonar import AfetsonarPipeline
        pipeline = AfetsonarPipeline(student_ckpt, device="cpu")

        # 100x100 px destroyed square centred in a 512x512 mask
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[200:300, 150:250] = 4  # destroyed

        bbox = (41.000, 28.970, 41.010, 28.980)  # ~1.11 km x ~0.84 km
        buildings = pipeline.mask_to_buildings(mask, bbox_latlon=bbox)

        assert len(buildings) == 1
        b = buildings[0]
        assert b["damage_class_name"] == "destroyed"

        # Footprint polygon: present, >= 4 vertices, all inside the bbox
        poly = b["polygon_latlon"]
        assert len(poly) >= 4
        for lat, lon in poly:
            assert bbox[0] <= lat <= bbox[2]
            assert bbox[1] <= lon <= bbox[3]

        # Area: 100x100 px of a 512-px-wide bbox
        # px_h = 111320*0.010/512 = 2.174 m ; px_w = 111320*cos(41.005)*0.010/512 = 1.641 m
        # expected ~ (100*100) * 2.174 * 1.641 = ~35,680 m2 (contourArea is
        # slightly smaller than the pixel count for filled shapes)
        assert 30_000 < b["area_m2"] < 40_000

        # Centroid must sit at the square's centre
        assert b["lat"] == pytest.approx(41.010 - (250 / 512) * 0.010, abs=1e-4)
        assert b["lon"] == pytest.approx(28.970 + (200 / 512) * 0.010, abs=1e-4)

    def test_buildings_to_geojson_polygons(self, student_ckpt):
        from afetsonar import AfetsonarPipeline
        from afetsonar.geo.utils import buildings_to_geojson

        pipeline = AfetsonarPipeline(student_ckpt, device="cpu")
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[50:100, 50:100] = 4
        mask[150:200, 120:180] = 2

        bbox = (41.000, 28.970, 41.010, 28.980)
        buildings = pipeline.mask_to_buildings(mask, bbox_latlon=bbox)
        gj = buildings_to_geojson(buildings, bbox_latlon=bbox)

        assert gj["type"] == "FeatureCollection"
        assert len(gj["features"]) == 2
        assert gj["bbox"] == [28.970, 41.000, 28.980, 41.010]  # [w, s, e, n]
        for feat in gj["features"]:
            geom = feat["geometry"]
            assert geom["type"] == "Polygon"
            ring = geom["coordinates"][0]
            assert ring[0] == ring[-1]  # closed ring
            for lon, lat in ring:      # GeoJSON order: [lon, lat]
                assert 28.970 <= lon <= 28.980
                assert 41.000 <= lat <= 41.010
            assert "damage_class_name" in feat["properties"]
