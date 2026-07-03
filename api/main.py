"""AFETSONAR REST API — feature-based endpoints for web integration.

Each capability is exposed as its own endpoint so a website can call
exactly what it needs:

===================  ========================================================
``GET  /health``     Liveness probe + model state.
``GET  /model-info`` Loaded model metadata (type, params, classes).
``POST /exif-gps``   Real-world location: GPS from the uploaded image's EXIF.
``POST /predict``    Damage mask only (base64 PNG + per-class stats).
``POST /buildings``  Geo-referenced buildings with FEMA priority scores.
``POST /map``        Interactive Folium map (full HTML or JSON-wrapped).
``POST /routes``     Team assignment + damage-weighted A* routes (JSON body).
``POST /analyze``    Full pipeline in one call (backward compatible).
===================  ========================================================

Usage::

    pip install fastapi uvicorn python-multipart pillow
    pip install -e ..   # install afetsonar package
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Interactive docs: http://<server>:8000/docs
"""

from __future__ import annotations

import io
import os
import tempfile
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

API_VERSION = "2.0.0"

app = FastAPI(
    title="AFETSONAR API",
    description=(
        "Global disaster damage assessment — drone/satellite imagery. "
        "Feature-based endpoints: /exif-gps, /predict, /buildings, /map, "
        "/routes, /analyze."
    ),
    version=API_VERSION,
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
        raise HTTPException(status_code=503, detail=_PIPELINE_ERR)

    for path in _CHECKPOINT_PATHS:
        if path and os.path.exists(path):
            try:
                from afetsonar import AfetsonarPipeline
                _PIPELINE = AfetsonarPipeline(path, device="auto")
                print(f"[AFETSONAR API] Model loaded: {path}")
                return _PIPELINE
            except Exception as exc:
                _PIPELINE_ERR = str(exc)
                raise HTTPException(status_code=503, detail=_PIPELINE_ERR)

    _PIPELINE_ERR = "Checkpoint not found. Set AFETSONAR_CHECKPOINT env var."
    raise HTTPException(status_code=503, detail=_PIPELINE_ERR)


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
    """Persist an upload to a temp file, preserving its extension.

    The original bytes are written untouched so EXIF metadata survives.
    """
    suffix = os.path.splitext(upload.filename or ".jpg")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(upload.file.read())
        return f.name


def _cleanup(*paths: Optional[str]) -> None:
    for p in paths:
        if p and os.path.exists(p):
            try:
                os.unlink(p)
            except OSError:
                pass


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


def _resolve_coords(
    post_tmp: str,
    lat: Optional[float],
    lon: Optional[float],
) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """Resolve coordinates: explicit form fields win, then EXIF GPS.

    Returns:
        ``(lat, lon, source)`` where source is ``"form"``, ``"exif"`` or
        ``None`` when no coordinates are available.
    """
    if lat is not None and lon is not None:
        return float(lat), float(lon), "form"

    from afetsonar.geo.utils import read_exif_gps
    exif = read_exif_gps(post_tmp)
    if exif:
        return exif["latitude"], exif["longitude"], "exif"
    return None, None, None


def _resolve_bbox(
    lat: Optional[float],
    lon: Optional[float],
    lat_min: Optional[float],
    lon_min: Optional[float],
    lat_max: Optional[float],
    lon_max: Optional[float],
    margin: float = 0.005,
) -> Optional[Tuple[float, float, float, float]]:
    """Explicit bbox wins; otherwise derive a ±margin° box around lat/lon."""
    if all(v is not None for v in [lat_min, lon_min, lat_max, lon_max]):
        return (float(lat_min), float(lon_min), float(lat_max), float(lon_max))
    if lat is not None and lon is not None:
        return (lat - margin, lon - margin, lat + margin, lon + margin)
    return None


def _serialize_buildings(buildings: List[Dict]) -> List[Dict]:
    out = []
    for b in buildings:
        polygon = b.get("polygon_latlon")
        out.append({
            "building_id":       b.get("building_id"),
            "damage_class":      b.get("damage_class"),
            "damage_class_name": b.get("damage_class_name"),
            "area_m2":           round(b.get("area_m2", 0), 1),
            "priority_score":    round(b.get("priority_score", 0), 3),
            "team_id":           b.get("team_id"),
            "lat":               b.get("lat"),
            "lon":               b.get("lon"),
            # Building outline as [lat, lon] vertices (Leaflet polygon
            # order). For [lon, lat] GeoJSON, use format="geojson".
            "polygon_latlon":    polygon,
        })
    return out


def _run_prediction(
    pipeline: Any,
    post_tmp: str,
    pre_tmp: Optional[str],
    use_tta: bool,
) -> np.ndarray:
    """Predict a damage mask, optionally with 8-transform TTA."""
    if use_tta:
        from afetsonar.evaluation.tta import TTAWrapper
        return TTAWrapper(pipeline, n_augmentations=8).predict(
            post_tmp, pre_path=pre_tmp
        )
    return pipeline.predict(post_tmp, pre_path=pre_tmp)


# ── Routes: request models ─────────────────────────────────────────────────────

class BuildingIn(BaseModel):
    """A geo-referenced building, as returned by ``/buildings``."""
    building_id: int
    lat: float
    lon: float
    damage_class: int = 1
    damage_class_name: str = "no_damage"
    area_m2: float = 0.0
    priority_score: float = 0.0


class RoutesRequest(BaseModel):
    """Request body for ``POST /routes``."""
    buildings: List[BuildingIn]
    bbox: Tuple[float, float, float, float] = Field(
        description="(lat_min, lon_min, lat_max, lon_max)"
    )
    n_teams: int = 3
    hospitals: List[Dict[str, Any]] = Field(default_factory=list)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness probe."""
    return {
        "status": "ok",
        "version": API_VERSION,
        "model_loaded": _PIPELINE is not None,
    }


@app.get("/model-info")
def model_info():
    """Metadata about the loaded model."""
    pipeline = _get_pipeline()
    model = pipeline.model
    n_params = sum(p.numel() for p in model.parameters())
    return {
        "model_class": type(model).__name__,
        "parameters": n_params,
        "parameters_million": round(n_params / 1e6, 1),
        "device": str(pipeline.device),
        "image_size": pipeline.config.image_size,
        "classes": _CLASS_NAMES,
        "api_version": API_VERSION,
    }


@app.post("/exif-gps")
async def exif_gps(image: UploadFile = File(...)):
    """Extract real-world GPS coordinates from an image's EXIF metadata.

    Send the drone's *original* JPEG — re-encoded/PNG images have no EXIF.
    """
    tmp = None
    try:
        tmp = _upload_to_tmp(image)
        from afetsonar.geo.utils import read_exif_gps
        coords = read_exif_gps(tmp)
        if coords is None:
            return JSONResponse({
                "success": True,
                "found": False,
                "detail": (
                    "No GPS EXIF data. PNGs and re-encoded images lose "
                    "metadata — upload the original camera JPEG."
                ),
            })
        return JSONResponse({
            "success": True,
            "found": True,
            "lat": coords["latitude"],
            "lon": coords["longitude"],
            "altitude_m": coords.get("altitude"),
        })
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        _cleanup(tmp)


@app.post("/predict")
async def predict(
    post_image: UploadFile = File(...),
    pre_image:  Optional[UploadFile] = File(None),
    use_tta:    bool = Form(False),
):
    """Damage segmentation only: colorized mask + per-class statistics."""
    post_tmp = pre_tmp = None
    pipeline = _get_pipeline()
    try:
        post_tmp = _upload_to_tmp(post_image)
        pre_tmp = _upload_to_tmp(pre_image) if pre_image else None

        mask = _run_prediction(pipeline, post_tmp, pre_tmp, use_tta)

        return JSONResponse({
            "success": True,
            "mask_png_b64": _mask_to_base64_png(mask),
            "mask_width":   int(mask.shape[1]),
            "mask_height":  int(mask.shape[0]),
            "stats":        _mask_stats(mask),
            "tta":          use_tta,
        })
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        _cleanup(post_tmp, pre_tmp)


@app.post("/buildings")
async def buildings_endpoint(
    post_image: UploadFile = File(...),
    pre_image:  Optional[UploadFile] = File(None),
    lat:        Optional[float] = Form(None),
    lon:        Optional[float] = Form(None),
    lat_min:    Optional[float] = Form(None),
    lon_min:    Optional[float] = Form(None),
    lat_max:    Optional[float] = Form(None),
    lon_max:    Optional[float] = Form(None),
    use_tta:    bool = Form(False),
    format:     str = Form("json"),
):
    """Detect buildings and score them with the FEMA priority formula.

    Coordinates: explicit ``lat``/``lon`` (or full bbox) win; otherwise
    the image's EXIF GPS is used automatically.

    ``format="geojson"`` returns an RFC 7946 FeatureCollection whose
    Polygon features are the building outlines — feed it directly to
    ``L.geoJSON(...)`` in Leaflet to draw the boundaries correctly.
    """
    post_tmp = pre_tmp = None
    pipeline = _get_pipeline()
    try:
        post_tmp = _upload_to_tmp(post_image)
        pre_tmp = _upload_to_tmp(pre_image) if pre_image else None

        lat, lon, coord_source = _resolve_coords(post_tmp, lat, lon)
        bbox = _resolve_bbox(lat, lon, lat_min, lon_min, lat_max, lon_max)

        mask = _run_prediction(pipeline, post_tmp, pre_tmp, use_tta)

        from afetsonar.routing.priority import score_buildings
        blds = pipeline.mask_to_buildings(mask, bbox_latlon=bbox)
        blds = score_buildings(blds)

        if format == "geojson":
            if bbox is None:
                raise HTTPException(
                    status_code=422,
                    detail="GeoJSON needs coordinates: pass lat/lon or a "
                           "bbox, or upload an image with GPS EXIF.",
                )
            from afetsonar.geo.utils import buildings_to_geojson
            return JSONResponse(
                buildings_to_geojson(blds, bbox_latlon=bbox),
                media_type="application/geo+json",
            )

        return JSONResponse({
            "success":      True,
            "buildings":    _serialize_buildings(blds),
            "n_buildings":  len(blds),
            "bbox":         list(bbox) if bbox else None,
            "coord_source": coord_source,
            "georeferenced": bbox is not None,
        })
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        _cleanup(post_tmp, pre_tmp)


@app.post("/map")
async def map_endpoint(
    post_image:      UploadFile = File(...),
    pre_image:       Optional[UploadFile] = File(None),
    lat:             Optional[float] = Form(None),
    lon:             Optional[float] = Form(None),
    lat_min:         Optional[float] = Form(None),
    lon_min:         Optional[float] = Form(None),
    lat_max:         Optional[float] = Form(None),
    lon_max:         Optional[float] = Form(None),
    hospitals_json:  str = Form("[]"),
    n_teams:         int = Form(3),
    include_routes:  bool = Form(True),
    include_lz:      bool = Form(True),
    response_format: str = Form("html"),
):
    """Generate the full interactive Folium map.

    ``response_format="html"`` returns ready-to-embed HTML (use it in an
    ``<iframe srcdoc>`` or serve it directly); ``"json"`` wraps the HTML
    in ``{"html": ...}``.

    ``hospitals_json``: JSON array of ``{"name", "lat", "lon"}`` objects.
    """
    import json as _json

    post_tmp = pre_tmp = map_tmp = None
    pipeline = _get_pipeline()
    try:
        post_tmp = _upload_to_tmp(post_image)
        pre_tmp = _upload_to_tmp(pre_image) if pre_image else None

        lat, lon, coord_source = _resolve_coords(post_tmp, lat, lon)
        bbox = _resolve_bbox(lat, lon, lat_min, lon_min, lat_max, lon_max)
        if bbox is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Map needs coordinates: pass lat/lon or a full bbox, "
                    "or upload an image with GPS EXIF."
                ),
            )

        try:
            hospitals = _json.loads(hospitals_json or "[]")
            assert isinstance(hospitals, list)
        except Exception:
            raise HTTPException(
                status_code=422,
                detail='hospitals_json must be a JSON array like '
                       '[{"name": "H1", "lat": 41.0, "lon": 28.9}]',
            )

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as mf:
            map_tmp = mf.name

        pipeline.generate_map(
            post_tmp,
            bbox_latlon=bbox,
            hospitals=hospitals,
            pre_path=pre_tmp,
            output_path=map_tmp,
            n_teams=n_teams,
            include_routes=include_routes,
            include_lz=include_lz,
        )
        with open(map_tmp, encoding="utf-8") as f:
            html = f.read()

        if response_format == "json":
            return JSONResponse({
                "success": True,
                "html": html,
                "bbox": list(bbox),
                "coord_source": coord_source,
            })
        return HTMLResponse(content=html)
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        _cleanup(post_tmp, pre_tmp, map_tmp)


@app.post("/routes")
def routes_endpoint(body: RoutesRequest):
    """Team assignment + damage-weighted A* routes for given buildings.

    Feed it the ``buildings`` array from ``/buildings`` — useful when the
    website renders its own map (e.g. Leaflet/Mapbox) and only needs the
    route geometry.  Requires internet access for the OSM road download.
    """
    pipeline = _get_pipeline()
    try:
        from afetsonar.routing.team_assignment import assign_hospitals, assign_teams

        buildings = [b.model_dump() for b in body.buildings]
        buildings, teams = assign_teams(buildings, n_teams=body.n_teams)
        teams = assign_hospitals(teams, body.hospitals)

        route_error = None
        try:
            routes = pipeline.compute_team_routes(buildings, teams, tuple(body.bbox))
        except Exception as exc:
            routes, route_error = [], str(exc)

        return JSONResponse({
            "success":   True,
            "teams":     teams,
            "buildings": _serialize_buildings(buildings),
            "routes":    routes,
            "route_error": route_error,
        })
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


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
    use_tta:    bool = Form(False),
):
    """Full pipeline in one call: mask + stats + prioritised buildings.

    Backward compatible with API v1; additionally resolves coordinates
    from EXIF when lat/lon are omitted (see ``coord_source``).
    """
    post_tmp = pre_tmp = None
    pipeline = _get_pipeline()
    try:
        post_tmp = _upload_to_tmp(post_image)
        pre_tmp = _upload_to_tmp(pre_image) if pre_image else None

        lat, lon, coord_source = _resolve_coords(post_tmp, lat, lon)
        bbox = _resolve_bbox(lat, lon, lat_min, lon_min, lat_max, lon_max)
        center_lat = ((bbox[0] + bbox[2]) / 2) if bbox else (lat or 0.0)
        center_lon = ((bbox[1] + bbox[3]) / 2) if bbox else (lon or 0.0)

        mask = _run_prediction(pipeline, post_tmp, pre_tmp, use_tta)

        from afetsonar.routing.priority import score_buildings
        blds = pipeline.mask_to_buildings(mask, bbox_latlon=bbox)
        blds = score_buildings(blds)

        geojson = None
        if bbox is not None:
            from afetsonar.geo.utils import buildings_to_geojson
            geojson = buildings_to_geojson(blds, bbox_latlon=bbox)

        return JSONResponse({
            "success":      True,
            "mask_png_b64": _mask_to_base64_png(mask),
            "mask_width":   int(mask.shape[1]),
            "mask_height":  int(mask.shape[0]),
            "stats":        _mask_stats(mask),
            "buildings":    _serialize_buildings(blds),
            "geojson":      geojson,
            "center_lat":   center_lat,
            "center_lon":   center_lon,
            "bbox":         list(bbox) if bbox else None,
            "coord_source": coord_source,
            "tta":          use_tta,
        })
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        _cleanup(post_tmp, pre_tmp)
