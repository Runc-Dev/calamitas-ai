"""Integration tests for AfetsonarPipeline.

These tests run on CPU without real checkpoints by creating a temporary
model file containing a minimal StudentSiameseSegformer state dict.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch


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
