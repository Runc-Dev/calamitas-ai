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


def test_analyze_end_to_end(app_module):
    rng = np.random.default_rng(42)
    post = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)

    (status, pre_out, post_out, mask_out, overlay_out,
     stats_md, buildings_df, map_html, map_file) = app_module.analyze(
        post_img=post,
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
