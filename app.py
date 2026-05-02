#!/usr/bin/env python3
"""AFETSONAR Gradio web application — Teknofest 2025.

Drone / satellite görüntülerinden afet hasar tespiti ve kurtarma rota planlaması.

HuggingFace Spaces deployment:
    - SDK: gradio
    - GPU: T4 (recommended)
    - AFETSONAR_CHECKPOINT env var → checkpoint path inside the Space

Local dev:
    pip install gradio
    AFETSONAR_CHECKPOINT=checkpoints/student_v1_best_ema.pth python app.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import pandas as pd

# ============================================================
# Lazy model management
# ============================================================

_PIPELINE: Any = None        # AfetsonarPipeline instance (loaded on first use)
_PIPELINE_ERR: str = ""      # Error message if loading failed

_CHECKPOINT_SEARCH_PATHS = [
    os.environ.get("AFETSONAR_CHECKPOINT", ""),
    "checkpoints/student_v1_best_ema.pth",
    "checkpoints/student/student_v1_best_ema.pth",
    "models/student_v1_best_ema.pth",
]


def _get_pipeline() -> Tuple[Any, str]:
    """Return (pipeline, error_msg).  Model is loaded at most once."""
    global _PIPELINE, _PIPELINE_ERR

    if _PIPELINE is not None:
        return _PIPELINE, ""
    if _PIPELINE_ERR:
        return None, _PIPELINE_ERR

    for path in _CHECKPOINT_SEARCH_PATHS:
        if path and os.path.exists(path):
            try:
                from afetsonar import AfetsonarPipeline
                _PIPELINE = AfetsonarPipeline(path, device="auto")
                print(f"[AFETSONAR] Model yüklendi: {path}")
                return _PIPELINE, ""
            except Exception as exc:
                _PIPELINE_ERR = f"Model yüklenemedi ({path}): {exc}"
                return None, _PIPELINE_ERR

    _PIPELINE_ERR = (
        "⚠️  Model checkpoint bulunamadı.\n"
        "Lütfen AFETSONAR_CHECKPOINT ortam değişkenini ayarlayın\n"
        "veya checkpoints/student_v1_best_ema.pth dosyasını koyun."
    )
    return None, _PIPELINE_ERR


# ============================================================
# Colour palette (6 damage classes)
# ============================================================

_DAMAGE_COLORS_RGB = np.array([
    [0,   0,   0  ],  # 0 background   — black
    [0,   200, 0  ],  # 1 no_damage    — green
    [255, 230, 0  ],  # 2 minor_damage — yellow
    [255, 128, 0  ],  # 3 major_damage — orange
    [220, 0,   0  ],  # 4 destroyed    — red
    [128, 0,   128],  # 5 unclassified — purple
], dtype=np.uint8)

_DAMAGE_LABELS_TR = ["arka plan", "sağlam", "az hasar", "ağır hasar", "yıkık", "belirsiz"]
_DAMAGE_LABELS_EN = ["background", "no_damage", "minor_damage", "major_damage", "destroyed", "unclassified"]


# ============================================================
# Visualization helpers
# ============================================================

def _colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert (H, W) uint8 mask → (H, W, 3) RGB colour image."""
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx in range(len(_DAMAGE_COLORS_RGB)):
        out[mask == cls_idx] = _DAMAGE_COLORS_RGB[cls_idx]
    return out


def _overlay_mask(img: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend damage mask colours onto the post-disaster image."""
    overlay = img.astype(np.float32).copy()
    for cls_idx in range(1, len(_DAMAGE_COLORS_RGB)):  # skip background
        region = mask == cls_idx
        if not region.any():
            continue
        color = _DAMAGE_COLORS_RGB[cls_idx].astype(np.float32)
        overlay[region] = (1 - alpha) * overlay[region] + alpha * color
    return overlay.clip(0, 255).astype(np.uint8)


def _mask_stats_markdown(mask: np.ndarray) -> str:
    """Return a Markdown table with per-class pixel fractions."""
    total = mask.size
    lines = [
        "### 📊 Hasar Dağılımı",
        "",
        "| Sınıf | EN | Piksel | % |",
        "|-------|----|---------|----|",
    ]
    for i, (tr, en) in enumerate(zip(_DAMAGE_LABELS_TR, _DAMAGE_LABELS_EN)):
        count = int((mask == i).sum())
        pct = 100.0 * count / max(total, 1)
        if i == 0 and pct > 99:
            continue  # skip background-only images
        lines.append(f"| {tr} | {en} | {count:,} | {pct:.1f}% |")
    return "\n".join(lines)


def _buildings_to_dataframe(buildings: List[Dict]) -> pd.DataFrame:
    cols = ["building_id", "damage_class_name", "area_m2",
            "priority_score", "lat", "lon"]
    rows = []
    for b in buildings:
        rows.append({
            "building_id":       b.get("building_id", "?"),
            "damage_class_name": b.get("damage_class_name", "?"),
            "area_m2":           round(b.get("area_m2", 0), 1),
            "priority_score":    round(b.get("priority_score", 0), 2),
            "lat":               round(b.get("lat", 0), 6) if "lat" in b else None,
            "lon":               round(b.get("lon", 0), 6) if "lon" in b else None,
        })
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)


def _generate_map_html(
    buildings: List[Dict],
    hospitals: List[Dict],
    center_lat: float,
    center_lon: float,
) -> str:
    """Build Folium map and return its full HTML as a string."""
    try:
        from afetsonar.geo.map_builder import FoliumMapBuilder
        from afetsonar.routing.team_assignment import assign_hospitals, assign_teams

        builder = FoliumMapBuilder(center_lat, center_lon, zoom_start=15)
        if buildings:
            bldgs_with_teams, teams = assign_teams(buildings, n_teams=min(5, len(buildings)))
            if hospitals:
                teams = assign_hospitals(teams, hospitals)
            builder.add_damage_markers(bldgs_with_teams)
        if hospitals:
            builder.add_hospitals(hospitals)
        builder.add_layer_control()
        return builder.map._repr_html_()
    except ImportError:
        return "<p>⚠️ Harita oluşturmak için <code>folium</code> gerekli.</p>"
    except Exception as exc:
        return f"<p>⚠️ Harita hatası: {exc}</p>"


def _save_array_to_tmp(arr: np.ndarray, suffix: str = ".png") -> str:
    """Save a numpy RGB image to a temp file and return the path."""
    from PIL import Image
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        Image.fromarray(arr).save(f.name)
        return f.name


# ============================================================
# Core event handlers
# ============================================================

def extract_exif_gps(post_img: Optional[np.ndarray]) -> Tuple[Any, Any, str]:
    """Read GPS coordinates from the uploaded post-disaster image's EXIF."""
    if post_img is None:
        return gr.update(), gr.update(), "❌ Önce görüntü yükleyin."

    try:
        from afetsonar.geo.auto_fetch import AutoPreFetcher
        tmp = _save_array_to_tmp(post_img, suffix=".png")
        coords = AutoPreFetcher().extract_gps(tmp)
        os.unlink(tmp)
    except Exception as exc:
        return gr.update(), gr.update(), f"⚠️ EXIF okunamadı: {exc}"

    if coords:
        msg = f"✅ GPS bulundu: {coords['lat']:.5f}°, {coords['lon']:.5f}°"
        return coords["lat"], coords["lon"], msg
    return gr.update(), gr.update(), "ℹ️ Bu görüntüde GPS EXIF verisi yok."


def add_hospital(
    name: str,
    h_lat: Optional[float],
    h_lon: Optional[float],
    hospitals: List[Dict],
) -> Tuple[List[Dict], pd.DataFrame, str, Any, Any]:
    """Append a hospital to the state list."""
    if not name or h_lat is None or h_lon is None:
        return hospitals, _hospitals_df(hospitals), "", gr.update(), gr.update()
    hospitals = hospitals + [{"name": name.strip(), "lat": float(h_lat), "lon": float(h_lon)}]
    return hospitals, _hospitals_df(hospitals), "", None, None


def clear_hospitals(hospitals: List[Dict]) -> Tuple[List[Dict], pd.DataFrame]:
    return [], _hospitals_df([])


def _hospitals_df(hospitals: List[Dict]) -> pd.DataFrame:
    if not hospitals:
        return pd.DataFrame(columns=["name", "lat", "lon"])
    return pd.DataFrame(hospitals)


def analyze(
    post_img:  Optional[np.ndarray],
    pre_img:   Optional[np.ndarray],
    lat:       Optional[float],
    lon:       Optional[float],
    lat_min:   Optional[float],
    lon_min:   Optional[float],
    lat_max:   Optional[float],
    lon_max:   Optional[float],
    provider:  str,
    api_key:   str,
    hospitals: List[Dict],
) -> Tuple[str, Any, Any, Any, Any, str, pd.DataFrame, str, Optional[str]]:
    """Run the full AFETSONAR pipeline and return all outputs.

    Returns
    -------
    status, pre_out, post_out, mask_out, overlay_out,
    stats_md, buildings_df, map_html, map_file
    """
    # ---- Guard: model must be loaded ----
    pipeline, err = _get_pipeline()
    if pipeline is None:
        return err, None, None, None, None, "", pd.DataFrame(), "", None

    if post_img is None:
        return "❌ Post-disaster görüntüsü gerekli.", None, None, None, None, "", pd.DataFrame(), "", None

    # ---- Save post image to tmp for pipeline ----
    post_tmp = _save_array_to_tmp(post_img)

    # ---- Resolve pre image ----
    pre_array: Optional[np.ndarray] = None
    status_parts: List[str] = []

    if pre_img is not None:
        pre_array = pre_img
        status_parts.append("✅ Pre görüntüsü: manuel yüklendi")
    elif api_key and lat is not None and lon is not None:
        try:
            from afetsonar.geo.auto_fetch import AutoPreFetcher
            fetcher = AutoPreFetcher(provider=provider, api_key=api_key.strip())
            pre_array = fetcher.fetch(lat=lat, lon=lon)
            status_parts.append(f"✅ Pre görüntüsü: {provider} API'den çekildi ({lat:.4f}, {lon:.4f})")
        except Exception as exc:
            status_parts.append(f"⚠️ Auto-fetch başarısız: {exc}. Post görüntüsü yedek olarak kullanıldı.")
    elif lat is not None and lon is not None:
        status_parts.append("ℹ️ API anahtarı yok — pre görüntüsü olarak post kullanıldı.")
    else:
        status_parts.append("ℹ️ Pre görüntüsü yok — post görüntüsü yedek olarak kullanıldı.")

    # ---- Save pre to tmp if array ----
    pre_tmp: Optional[str] = None
    if pre_array is not None:
        pre_tmp = _save_array_to_tmp(pre_array)

    # ---- Run prediction ----
    try:
        mask = pipeline.predict(post_tmp, pre_path=pre_tmp)
        status_parts.append("✅ Hasar maskesi oluşturuldu.")
    except Exception as exc:
        _cleanup_tmps(post_tmp, pre_tmp)
        return f"❌ Tahmin hatası: {exc}", None, None, None, None, "", pd.DataFrame(), "", None

    # ---- Resolve bbox ----
    bbox: Optional[Tuple[float, float, float, float]] = None
    center_lat = lat or 0.0
    center_lon = lon or 0.0

    if all(v is not None for v in [lat_min, lon_min, lat_max, lon_max]):
        bbox = (float(lat_min), float(lon_min), float(lat_max), float(lon_max))
        center_lat = (lat_min + lat_max) / 2
        center_lon = (lon_min + lon_max) / 2
        status_parts.append("✅ Coğrafi bbox: harita oluşturulacak.")
    elif lat is not None and lon is not None:
        margin = 0.005  # ±0.005° ≈ 550 m
        bbox = (lat - margin, lon - margin, lat + margin, lon + margin)
        center_lat, center_lon = lat, lon
        status_parts.append("ℹ️ Bbox GPS noktasından otomatik türetildi.")

    # ---- Building extraction ----
    try:
        from afetsonar.routing.priority import score_buildings
        buildings = pipeline.mask_to_buildings(mask, bbox_latlon=bbox)
        buildings = score_buildings(buildings)
        status_parts.append(f"✅ {len(buildings)} bina tespit edildi.")
    except Exception as exc:
        buildings = []
        status_parts.append(f"⚠️ Bina çıkarma hatası: {exc}")

    # ---- Visualize ----
    # Resize mask to match post image for display
    h_orig, w_orig = post_img.shape[:2]
    from PIL import Image as PILImage
    mask_resized = np.array(
        PILImage.fromarray(mask).resize((w_orig, h_orig), PILImage.NEAREST)
    )

    mask_color   = _colorize_mask(mask_resized)
    overlay_img  = _overlay_mask(post_img, mask_resized, alpha=0.5)
    pre_display  = pre_array if pre_array is not None else post_img

    stats_md   = _mask_stats_markdown(mask_resized)
    bldgs_df   = _buildings_to_dataframe(buildings)

    # ---- Generate map HTML ----
    map_html_content = ""
    map_file_path: Optional[str] = None
    if bbox is not None and (buildings or hospitals):
        map_html_content = _generate_map_html(
            buildings, hospitals, center_lat, center_lon
        )
        # Save map to temp file for download
        with tempfile.NamedTemporaryFile(
            suffix=".html", prefix="afetsonar_map_", delete=False
        ) as mf:
            map_file_path = mf.name
            mf.write(map_html_content.encode())
        status_parts.append("✅ İnteraktif harita oluşturuldu.")

    _cleanup_tmps(post_tmp, pre_tmp)

    status = "\n".join(status_parts)
    wrapped_map = (
        f'<div style="width:100%;height:600px;border:1px solid #ccc;">'
        f'{map_html_content}'
        f'</div>'
        if map_html_content else
        "<p style='color:#888;text-align:center'>Harita için GPS koordinatı veya bbox gerekli.</p>"
    )

    return (
        status,
        pre_display,    # pre_out
        post_img,       # post_out
        mask_color,     # mask_out
        overlay_img,    # overlay_out
        stats_md,
        bldgs_df,
        wrapped_map,
        map_file_path,
    )


def _cleanup_tmps(*paths: Optional[str]) -> None:
    for p in paths:
        if p and os.path.exists(p):
            try:
                os.unlink(p)
            except OSError:
                pass


# ============================================================
# Gradio UI
# ============================================================

_APP_CSS = """
    .header-logo { font-size: 2em; font-weight: 900; color: #c62828; }
    .label-badge {
        display: inline-block; padding: 2px 8px; border-radius: 12px;
        font-size: 0.78em; font-weight: bold;
    }
"""
_APP_THEME = gr.themes.Soft(primary_hue="red", secondary_hue="orange")


def build_ui() -> gr.Blocks:
    # Gradio 6+ moved theme/css from Blocks() to launch(); keep both for compatibility
    _gr_major = int(gr.__version__.split(".")[0])
    blocks_kwargs: dict = {"title": "AFETSONAR — Afet Hasar Degerlendirme"}
    if _gr_major < 6:
        blocks_kwargs["theme"] = _APP_THEME
        blocks_kwargs["css"] = _APP_CSS

    with gr.Blocks(**blocks_kwargs) as demo:
        # ---- Header ----
        gr.Markdown(
            """
            # 🏚️ AFETSONAR — Afet Hasar Değerlendirme Sistemi
            **Drone / uydu görüntülerinden bina hasar tespiti ve kurtarma rota planlaması.**
            Teknofest 2025 · SegFormer-B0 Siamese · 36ms/görüntü · mF1 0.617
            ---
            """
        )

        hospitals_state = gr.State([])

        with gr.Row(equal_height=False):
            # ----------------------------------------------------------------
            # LEFT COLUMN — Inputs
            # ----------------------------------------------------------------
            with gr.Column(scale=1, min_width=340):
                gr.Markdown("### 📥 Girdi")

                post_img = gr.Image(
                    label="Post-disaster görüntü *",
                    type="numpy",
                    sources=["upload", "clipboard"],
                )
                pre_img = gr.Image(
                    label="Pre-disaster görüntü (opsiyonel)",
                    type="numpy",
                    sources=["upload", "clipboard"],
                )

                # GPS
                with gr.Accordion("📍 GPS Koordinatları", open=True):
                    exif_btn = gr.Button(
                        "🔍 EXIF'ten Oku", variant="secondary", size="sm"
                    )
                    exif_status = gr.Textbox(
                        label="EXIF durumu", interactive=False,
                        placeholder="Görüntü yükleyip butona basın…",
                    )
                    with gr.Row():
                        lat = gr.Number(label="Enlem (lat)", precision=6)
                        lon = gr.Number(label="Boylam (lon)", precision=6)

                # Bounding box
                with gr.Accordion("🗺️ Harita Sınırları (opsiyonel)", open=False):
                    gr.Markdown(
                        "_Harita oluşturmak için doldurun. Boş bırakırsanız GPS'ten otomatik türetilir._"
                    )
                    with gr.Row():
                        lat_min = gr.Number(label="lat_min", precision=6)
                        lon_min = gr.Number(label="lon_min", precision=6)
                    with gr.Row():
                        lat_max = gr.Number(label="lat_max", precision=6)
                        lon_max = gr.Number(label="lon_max", precision=6)

                # Auto-fetch
                with gr.Accordion("🛰️ Uydu Pre-Görüntüsü Çekme (opsiyonel)", open=False):
                    gr.Markdown(
                        "_GPS + API anahtarı varsa pre-disaster görüntüsü otomatik indirilir._"
                    )
                    provider = gr.Dropdown(
                        choices=["google", "mapbox"],
                        value="google",
                        label="Sağlayıcı",
                    )
                    api_key = gr.Textbox(
                        label="API Anahtarı",
                        type="password",
                        placeholder="Google Maps Key veya Mapbox Token",
                    )

                # Hospitals
                with gr.Accordion("🏥 Hastaneler / Toplanma Noktaları (opsiyonel)", open=False):
                    hosp_name = gr.Textbox(label="Kurum adı")
                    with gr.Row():
                        hosp_lat = gr.Number(label="Enlem", precision=6)
                        hosp_lon = gr.Number(label="Boylam", precision=6)
                    with gr.Row():
                        add_hosp_btn  = gr.Button("➕ Ekle", size="sm")
                        clr_hosp_btn  = gr.Button("🗑️ Temizle", size="sm", variant="stop")
                    hosp_table = gr.Dataframe(
                        headers=["name", "lat", "lon"],
                        datatype=["str", "number", "number"],
                        label="Hastane listesi",
                        interactive=False,
                    )

                analyze_btn = gr.Button(
                    "🔍 Analiz Et", variant="primary", size="lg"
                )

            # ----------------------------------------------------------------
            # RIGHT COLUMN — Outputs
            # ----------------------------------------------------------------
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Sonuçlar")

                status_box = gr.Textbox(
                    label="Durum", lines=4, interactive=False,
                    placeholder="Analiz çalıştırdıktan sonra sonuçlar burada görünür…",
                )

                with gr.Row():
                    out_pre     = gr.Image(label="🛰️ Afet öncesi",   type="numpy")
                    out_post    = gr.Image(label="📷 Afet sonrası",   type="numpy")
                with gr.Row():
                    out_mask    = gr.Image(label="🎨 Hasar maskesi", type="numpy")
                    out_overlay = gr.Image(label="🔍 Overlay",        type="numpy")

                stats_md = gr.Markdown()

                out_buildings = gr.Dataframe(
                    label="🏗️ Tespit Edilen Binalar",
                    wrap=True,
                )

                out_map = gr.HTML(
                    label="🗺️ İnteraktif Harita",
                    value="<p style='color:#888;text-align:center;padding:40px;'>"
                          "Analiz çalıştırın…</p>",
                )
                out_map_file = gr.File(
                    label="⬇️ Haritayı İndir (.html)",
                    visible=True,
                )

        # ---- Legend ----
        gr.Markdown(
            """
            ---
            **Renk Kodu:**
            🟩 Sağlam &nbsp; 🟨 Az hasar &nbsp; 🟧 Ağır hasar &nbsp; 🟥 Yıkık &nbsp; 🟪 Belirsiz
            """
        )

        # ================================================================
        # Event wiring
        # ================================================================

        exif_btn.click(
            fn=extract_exif_gps,
            inputs=[post_img],
            outputs=[lat, lon, exif_status],
        )

        add_hosp_btn.click(
            fn=add_hospital,
            inputs=[hosp_name, hosp_lat, hosp_lon, hospitals_state],
            outputs=[hospitals_state, hosp_table, hosp_name, hosp_lat, hosp_lon],
        )

        clr_hosp_btn.click(
            fn=clear_hospitals,
            inputs=[hospitals_state],
            outputs=[hospitals_state, hosp_table],
        )

        analyze_btn.click(
            fn=analyze,
            inputs=[
                post_img, pre_img,
                lat, lon,
                lat_min, lon_min, lat_max, lon_max,
                provider, api_key,
                hospitals_state,
            ],
            outputs=[
                status_box,
                out_pre, out_post, out_mask, out_overlay,
                stats_md,
                out_buildings,
                out_map,
                out_map_file,
            ],
        )

    return demo


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    _gr_major = int(gr.__version__.split(".")[0])
    demo = build_ui()
    launch_kwargs: dict = {
        "server_name": "0.0.0.0",
        "share": False,
        "show_error": True,
    }
    if _gr_major >= 6:
        launch_kwargs["theme"] = _APP_THEME
        launch_kwargs["css"] = _APP_CSS
    demo.launch(**launch_kwargs)
