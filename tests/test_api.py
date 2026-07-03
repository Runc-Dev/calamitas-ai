"""Smoke tests for the FastAPI backend (``api/main.py``).

Uses a random-weight student checkpoint so no trained model or GPU
is required — mirrors the torch-optional style of the other tests.
"""

from __future__ import annotations

import importlib
import io
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

ROOT = Path(__file__).resolve().parents[1]


def _png_bytes(seed: int = 0, size: int = 128) -> bytes:
    """Return an in-memory random RGB PNG."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def api_client(tmp_path, monkeypatch):
    """TestClient backed by a random-weight student checkpoint."""
    from afetsonar.models import StudentSiameseSegformer

    ckpt = tmp_path / "student_random.pth"
    model = StudentSiameseSegformer(pretrained=False)
    torch.save({"model_state_dict": model.state_dict()}, str(ckpt))
    monkeypatch.setenv("AFETSONAR_CHECKPOINT", str(ckpt))

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import api.main as api_main

    # Reload so the module re-reads AFETSONAR_CHECKPOINT and resets
    # its lazy _PIPELINE / _PIPELINE_ERR globals.
    api_main = importlib.reload(api_main)

    from fastapi.testclient import TestClient

    return TestClient(api_main.app)


def test_health_endpoint(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_analyze_returns_mask_stats_and_buildings(api_client):
    files = {
        "post_image": ("post.png", _png_bytes(seed=1), "image/png"),
        "pre_image": ("pre.png", _png_bytes(seed=2), "image/png"),
    }
    data = {"lat": "41.005", "lon": "28.977"}

    resp = api_client.post("/analyze", files=files, data=data)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True
    assert body["mask_width"] > 0 and body["mask_height"] > 0
    assert len(body["mask_png_b64"]) > 0
    assert len(body["stats"]) == 6
    assert isinstance(body["buildings"], list)
    # lat/lon given → bbox must be derived around the GPS point
    assert body["bbox"] is not None
    assert body["center_lat"] == pytest.approx(41.005)


def test_analyze_without_post_image_fails(api_client):
    resp = api_client.post("/analyze", files={}, data={})
    # FastAPI rejects the request before our handler runs (missing file)
    assert resp.status_code == 422
