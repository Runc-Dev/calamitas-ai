"""Tests for ONNX export (``afetsonar/deployment.py``).

Uses a random-weight student model at a small resolution so the tests
run in seconds on CPU.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from afetsonar.deployment import export_to_onnx, verify_onnx  # noqa: E402


@pytest.fixture()
def student_ckpt(tmp_path):
    """Random-weight student checkpoint on disk."""
    from afetsonar.models import StudentSiameseSegformer

    ckpt = tmp_path / "student_random.pth"
    model = StudentSiameseSegformer(pretrained=False)
    torch.save({"model_state_dict": model.state_dict()}, str(ckpt))
    return str(ckpt)


def test_export_creates_valid_onnx(student_ckpt, tmp_path):
    out = tmp_path / "student.onnx"
    path = export_to_onnx(student_ckpt, out, image_size=128)
    assert out.exists()
    assert out.stat().st_size > 1_000_000  # 4.3M params ≈ 17 MB fp32
    assert path == str(out)


def test_onnxruntime_parity_within_tolerance(student_ckpt, tmp_path):
    out = tmp_path / "student.onnx"
    export_to_onnx(student_ckpt, out, image_size=128)
    max_diff = verify_onnx(student_ckpt, out, image_size=128, atol=1e-3)
    assert max_diff <= 1e-3


def test_dynamic_batch_axis(student_ckpt, tmp_path):
    import onnxruntime as ort

    out = tmp_path / "student.onnx"
    export_to_onnx(student_ckpt, out, image_size=128)

    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).standard_normal((2, 6, 128, 128)).astype(np.float32)
    (y,) = session.run(None, {"pre_post": x})

    assert y.shape[0] == 2       # batch axis is dynamic
    assert y.shape[1] == 6       # six damage classes
