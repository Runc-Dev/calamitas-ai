"""Smoke tests for the FastAPI backend (``api/main.py``).

Covers the feature-based endpoints (v2): /health, /model-info,
/exif-gps, /predict, /buildings, /map, /routes, /analyze.

Uses a random-weight student checkpoint so no trained model or GPU is
required — mirrors the torch-optional style of the other tests.
Network-dependent layers (OSM routes/LZ) are disabled in tests.
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


def _gps_jpeg_bytes(seed: int = 0, size: int = 96) -> bytes:
    """Return an in-memory JPEG with GPS EXIF (41.005 N, 28.977 E)."""
    from PIL import Image
    from PIL.TiffImagePlugin import IFDRational

    rng = np.random.default_rng(seed)
    img = Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8))
    exif = Image.Exif()
    exif[0x8825] = {
        1: "N",
        2: (IFDRational(41, 1), IFDRational(0, 1), IFDRational(18, 1)),
        3: "E",
        4: (IFDRational(28, 1), IFDRational(58, 1), IFDRational(372, 10)),
    }
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
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


# ---------------------------------------------------------------------------
# /health + /model-info
# ---------------------------------------------------------------------------

def test_health_endpoint(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_model_info(api_client):
    resp = api_client.get("/model-info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_class"] == "StudentSiameseSegformer"
    assert body["parameters_million"] > 1
    assert len(body["classes"]) == 6


# ---------------------------------------------------------------------------
# /exif-gps — real-world location extraction
# ---------------------------------------------------------------------------

def test_exif_gps_found(api_client):
    files = {"image": ("drone.jpg", _gps_jpeg_bytes(), "image/jpeg")}
    resp = api_client.post("/exif-gps", files=files)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["found"] is True
    assert body["lat"] == pytest.approx(41.005, abs=1e-3)
    assert body["lon"] == pytest.approx(28.977, abs=1e-3)


def test_exif_gps_not_found_in_png(api_client):
    files = {"image": ("plain.png", _png_bytes(), "image/png")}
    resp = api_client.post("/exif-gps", files=files)
    assert resp.status_code == 200
    body = resp.json()
    assert body["found"] is False
    assert "EXIF" in body["detail"] or "JPEG" in body["detail"]


# ---------------------------------------------------------------------------
# /predict — mask only
# ---------------------------------------------------------------------------

def test_predict_returns_mask_and_stats(api_client):
    files = {
        "post_image": ("post.png", _png_bytes(seed=1), "image/png"),
        "pre_image": ("pre.png", _png_bytes(seed=2), "image/png"),
    }
    resp = api_client.post("/predict", files=files)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True
    assert body["mask_width"] > 0
    assert len(body["stats"]) == 6
    assert body["tta"] is False


# ---------------------------------------------------------------------------
# /buildings — geo-referenced detections
# ---------------------------------------------------------------------------

def test_buildings_with_form_coords(api_client):
    files = {"post_image": ("post.png", _png_bytes(seed=3), "image/png")}
    data = {"lat": "41.005", "lon": "28.977"}
    resp = api_client.post("/buildings", files=files, data=data)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True
    assert body["coord_source"] == "form"
    assert body["georeferenced"] is True
    assert body["bbox"] is not None
    for b in body["buildings"]:
        assert b["lat"] is not None and b["lon"] is not None


def test_buildings_include_footprint_polygons(api_client):
    files = {"post_image": ("post.png", _png_bytes(seed=7), "image/png")}
    data = {"lat": "41.005", "lon": "28.977"}
    resp = api_client.post("/buildings", files=files, data=data)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    for b in body["buildings"]:
        poly = b["polygon_latlon"]
        assert poly is not None and len(poly) >= 3
        eps = 1e-9  # float arithmetic can overshoot the bbox edge by ~1e-15
        for lat, lon in poly:
            assert 41.0 - eps <= lat <= 41.01 + eps
            assert 28.972 - eps <= lon <= 28.982 + eps


def test_buildings_geojson_format(api_client):
    files = {"post_image": ("post.png", _png_bytes(seed=8), "image/png")}
    data = {"lat": "41.005", "lon": "28.977", "format": "geojson"}
    resp = api_client.post("/buildings", files=files, data=data)
    assert resp.status_code == 200, resp.text
    gj = resp.json()
    assert gj["type"] == "FeatureCollection"
    for feat in gj["features"]:
        assert feat["geometry"]["type"] in ("Polygon", "Point")
        if feat["geometry"]["type"] == "Polygon":
            ring = feat["geometry"]["coordinates"][0]
            assert ring[0] == ring[-1]  # closed ring, [lon, lat] order


def test_buildings_coords_from_exif(api_client):
    """No lat/lon fields — location must come from the image EXIF."""
    files = {"post_image": ("drone.jpg", _gps_jpeg_bytes(seed=4), "image/jpeg")}
    resp = api_client.post("/buildings", files=files)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["coord_source"] == "exif"
    assert body["georeferenced"] is True
    bbox = body["bbox"]
    assert bbox[0] < 41.005 < bbox[2]
    assert bbox[1] < 28.977 < bbox[3]


# ---------------------------------------------------------------------------
# /map — interactive Folium map
# ---------------------------------------------------------------------------

def test_map_json_format(api_client):
    files = {"post_image": ("post.png", _png_bytes(seed=5), "image/png")}
    data = {
        "lat": "41.005", "lon": "28.977",
        "hospitals_json": '[{"name": "Cerrahpasa", "lat": 41.0048, "lon": 28.951}]',
        "include_routes": "false",   # avoid OSM network in tests
        "include_lz": "false",
        "response_format": "json",
    }
    resp = api_client.post("/map", files=files, data=data)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["success"] is True
    assert "<html" in body["html"].lower() or "folium" in body["html"].lower()
    assert "Damage Assessment" in body["html"]


def test_map_requires_coordinates(api_client):
    files = {"post_image": ("post.png", _png_bytes(seed=6), "image/png")}
    resp = api_client.post(
        "/map", files=files,
        data={"include_routes": "false", "include_lz": "false"},
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /routes — team assignment + routing from building JSON
# ---------------------------------------------------------------------------

def test_routes_assigns_teams(api_client):
    buildings = [
        {"building_id": i, "lat": 41.004 + i * 0.0004, "lon": 28.976 + i * 0.0004,
         "damage_class": 4, "damage_class_name": "destroyed",
         "area_m2": 200.0, "priority_score": 5.0}
        for i in range(6)
    ]
    body = {
        "buildings": buildings,
        "bbox": [41.003, 28.975, 41.008, 28.981],
        "n_teams": 2,
        "hospitals": [{"name": "H1", "lat": 41.0048, "lon": 28.951}],
    }
    resp = api_client.post("/routes", json=body)
    assert resp.status_code == 200, resp.text
    out = resp.json()
    assert out["success"] is True
    assert 1 <= len(out["teams"]) <= 2
    for t in out["teams"]:
        assert t["assigned_hospital"] == "H1"
    for b in out["buildings"]:
        assert b["team_id"] is not None
    # routes may be empty when OSM is unreachable — the error is reported
    assert "routes" in out


# ---------------------------------------------------------------------------
# /analyze — full pipeline (backward compatible)
# ---------------------------------------------------------------------------

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
    assert body["bbox"] is not None
    assert body["center_lat"] == pytest.approx(41.005)
    assert body["coord_source"] == "form"


def test_analyze_without_post_image_fails(api_client):
    resp = api_client.post("/analyze", files={}, data={})
    assert resp.status_code == 422
