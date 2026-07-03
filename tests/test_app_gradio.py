"""Smoke tests for the Gradio web app (``app.py``).

Validates that the UI builds against the installed gradio version and
that the ``analyze`` callback runs end-to-end on a synthetic image with
a random-weight student checkpoint (no GPU, no network).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("gradio")

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def app_module(tmp_path, monkeypatch):
    """Import ``app`` with a random-weight student checkpoint configured."""
    from afetsonar.models import StudentSiameseSegformer

    ckpt = tmp_path / "student_random.pth"
    model = StudentSiameseSegformer(pretrained=False)
    torch.save({"model_state_dict": model.state_dict()}, str(ckpt))
    monkeypatch.setenv("AFETSONAR_CHECKPOINT", str(ckpt))

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import app

    # Reload so the module re-reads AFETSONAR_CHECKPOINT and resets
    # its lazy _PIPELINE / _PIPELINE_ERR globals.
    return importlib.reload(app)


def test_build_ui_constructs(app_module):
    """The Blocks UI must build without errors on the installed gradio."""
    demo = app_module.build_ui()
    import gradio as gr

    assert isinstance(demo, gr.Blocks)


def test_analyze_end_to_end(app_module, tmp_path):
    from PIL import Image

    rng = np.random.default_rng(42)
    post = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
    post_path = tmp_path / "post.png"
    Image.fromarray(post).save(post_path)

    (status, pre_out, post_out, mask_out, overlay_out,
     stats_md, buildings_df, map_html, map_file) = app_module.analyze(
        post_img=str(post_path),
        pre_img=None,
        lat=41.005,
        lon=28.977,
        lat_min=None,
        lon_min=None,
        lat_max=None,
        lon_max=None,
        provider="google",
        api_key="",
        hospitals=[],
        use_tta=False,
    )

    assert "❌" not in status, status
    assert mask_out is not None
    assert overlay_out is not None
    assert stats_md  # per-class statistics rendered


def test_analyze_reads_gps_from_exif(app_module, tmp_path):
    """When lat/lon are empty, coordinates must come from the image EXIF."""
    from PIL import Image
    from PIL.TiffImagePlugin import IFDRational

    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (96, 96, 3), dtype=np.uint8))
    exif = Image.Exif()
    exif[0x8825] = {
        1: "N",
        2: (IFDRational(41, 1), IFDRational(0, 1), IFDRational(18, 1)),
        3: "E",
        4: (IFDRational(28, 1), IFDRational(58, 1), IFDRational(372, 10)),
    }
    post_path = tmp_path / "drone.jpg"
    img.save(post_path, exif=exif)

    (status, *_rest) = app_module.analyze(
        post_img=str(post_path),
        pre_img=None,
        lat=None,
        lon=None,
        lat_min=None,
        lon_min=None,
        lat_max=None,
        lon_max=None,
        provider="google",
        api_key="",
        hospitals=[],
        use_tta=False,
    )

    assert "EXIF" in status and "41.005" in status, status


def test_extract_exif_gps_from_file(app_module, tmp_path):
    from PIL import Image
    from PIL.TiffImagePlugin import IFDRational

    rng = np.random.default_rng(1)
    img = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    exif = Image.Exif()
    exif[0x8825] = {
        1: "S",
        2: (IFDRational(33, 1), IFDRational(52, 1), IFDRational(0, 1)),
        3: "E",
        4: (IFDRational(151, 1), IFDRational(12, 1), IFDRational(0, 1)),
    }
    path = tmp_path / "sydney.jpg"
    img.save(path, exif=exif)

    lat, lon, msg = app_module.extract_exif_gps(str(path))
    assert "✅" in msg
    assert lat < 0  # southern hemisphere
    assert 151 < lon < 152
