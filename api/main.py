"""AFETSONAR REST API — global deployment backend.

Endpoints:
    POST /analyze   — damage assessment from pre/post image pair
    GET  /health    — liveness probe

Usage:
    pip install fastapi uvicorn python-multipart pillow
    pip install -e ..   # install afetsonar package
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Mobile app calls:
    POST http://<server>:8000/analyze
    Content-Type: multipart/form-data
    Fields: post_image (file), pre_image (file, optional),
            lat (float, optional), lon (float, optional),
            lat_min/lon_min/lat_max/lon_max (float, optional bbox)
"""

from __future__ import annotations

import io
import os
import tempfile
import traceback
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

app = FastAPI(
    title="AFETSONAR API",
    description="Global disaster damage assessment — drone/satellite imagery",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy pipeline ──────────────────────────────────────────────────────────────

_PIPELINE: Any = None
_PIPELINE_ERR: str = ""

_CHECKPOINT_PATHS = [
    os.environ.get("AFETSONAR_CHECKPOINT", ""),
    "checkpoints/student_v1_best_ema.pth",
    "checkpoints/student/student_v1_best_ema.pth",
    "../checkpoints/student/student_v1_best_ema.pth",
]


def _get_pipeline():
    global _PIPELINE, _PIPELINE_ERR
    if _PIPELINE is not None:
        return _PIPELINE
    if _PIPELINE_ERR:
        raise RuntimeError(_PIPELINE_ERR)

    for path in _CHECKPOINT_PATHS:
        if path and os.path.exists(path):
            try:
                from afetsonar import AfetsonarPipeline
                _PIPELINE = AfetsonarPipeline(path, device="auto")
                print(f"[AFETSONAR API] Model loaded: {path}")
                return _PIPELINE
            except Exception as exc:
                _PIPELINE_ERR = str(exc)
                raise RuntimeError(_PIPELINE_ERR)

    _PIPELINE_ERR = "Checkpoint not found. Set AFETSONAR_CHECKPOINT env var."
    raise RuntimeError(_PIPELINE_ERR)


# ── Helpers ────────────────────────────────────────────────────────────────────

_DAMAGE_COLORS = [
    [0,   0,   0  ],  # 0 background
    [0,   200, 0  ],  # 1 no_damage
    [255, 230, 0  ],  # 2 minor_damage
    [255, 128, 0  ],  # 3 major_damage
    [220, 0,   0  ],  # 4 destroyed
    [128, 0,   128],  # 5 unclassified
]

_CLASS_NAMES = ["background", "no_damage", "minor_damage",
                "major_damage", "destroyed", "unclassified"]


def _upload_to_tmp(upload: UploadFile) -> str:
    suffix = os.path.splitext(upload.filename or ".jpg")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(upload.file.read())
        return f.name


def _mask_to_base64_png(mask: np.ndarray) -> str:
    """Convert (H,W) uint8 mask to colorized PNG base64 string."""
    import base64
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, color in enumerate(_DAMAGE_COLORS):
        rgb[mask == cls_idx] = color
    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _mask_stats(mask: np.ndarray) -> List[Dict]:
    total = mask.size
    stats = []
    for i, name in enumerate(_CLASS_NAMES):
        count = int((mask == i).sum())
        stats.append({
            "class_id": i,
            "class_name": name,
            "pixel_count": count,
            "percentage": round(100.0 * count / max(total, 1), 2),
        })
    return stats


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _PIPELINE is not None}


@app.post("/analyze")
async def analyze(
    post_image: UploadFile = File(...),
    pre_image:  Optional[UploadFile] = File(None),
    lat:        Optional[float] = Form(None),
    lon:        Optional[float] = Form(None),
    lat_min:    Optional[float] = Form(None),
    lon_min:    Optional[float] = Form(None),
    lat_max:    Optional[float] = Form(None),
    lon_max:    Optional[float] = Form(None),
):
    """Run full AFETSONAR pipeline on uploaded images.

    Returns JSON with:
    - mask_png_b64: colorized damage mask as base64 PNG
    - stats: per-class pixel statistics
    - buildings: list of detected damaged buildings with lat/lon
    - center_lat, center_lon: map center coordinates
    - bbox: bounding box used (or null)
    """
    post_tmp = pre_tmp = None
    try:
        pipeline = _get_pipeline()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    try:
        post_tmp = _upload_to_tmp(post_image)
        pre_tmp  = _upload_to_tmp(pre_image) if pre_image else None

        # Run prediction
        mask = pipeline.predict(post_tmp, pre_path=pre_tmp, lat=lat, lon=lon)

        # Resolve bbox
        bbox = None
        center_lat = lat or 0.0
        center_lon = lon or 0.0

        if all(v is not None for v in [lat_min, lon_min, lat_max, lon_max]):
            bbox = (float(lat_min), float(lon_min), float(lat_max), float(lon_max))
            center_lat = (lat_min + lat_max) / 2
            center_lon = (lon_min + lon_max) / 2
        elif lat is not None and lon is not None:
            margin = 0.005
            bbox = (lat - margin, lon - margin, lat + margin, lon + margin)
            center_lat, center_lon = lat, lon

        # Extract buildings
        from afetsonar.routing.priority import score_buildings
        buildings = pipeline.mask_to_buildings(mask, bbox_latlon=bbox)
        buildings = score_buildings(buildings)

        # Serialize buildings (remove non-serializable keys)
        bldgs_out = []
        for b in buildings:
            bldgs_out.append({
                "building_id":       b.get("building_id"),
                "damage_class":      b.get("damage_class"),
                "damage_class_name": b.get("damage_class_name"),
                "area_m2":           round(b.get("area_m2", 0), 1),
                "priority_score":    round(b.get("priority_score", 0), 3),
                "lat":               b.get("lat"),
                "lon":               b.get("lon"),
            })

        return JSONResponse({
            "success":    True,
            "mask_png_b64": _mask_to_base64_png(mask),
            "mask_width":   int(mask.shape[1]),
            "mask_height":  int(mask.shape[0]),
            "stats":        _mask_stats(mask),
            "buildings":    bldgs_out,
            "center_lat":   center_lat,
            "center_lon":   center_lon,
            "bbox":         list(bbox) if bbox else None,
        })

    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        for p in [post_tmp, pre_tmp]:
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass
