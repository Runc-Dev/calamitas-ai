"""PyTorch <-> TensorFlow parity for the converted Siamese teacher.

Mirrors the repo's ONNX parity pattern (atol 1e-3, fp32, CPU): the
golden input/outputs were produced by ``scripts/export_weights_npz.py``
in the torch env; here the converted TF model must reproduce all six
output tensors.

Skipped automatically when the (gitignored) export artifacts are absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
pytest.importorskip("transformers")

ROOT = Path(__file__).resolve().parents[1]
EXPORT_DIR = ROOT / "checkpoints" / "export"

ATOL = 1e-3


@pytest.fixture(scope="module")
def converted_model():
    if not (EXPORT_DIR / "teacher_v4_ema_hf.npz").exists():
        pytest.skip("export npz missing — run scripts/export_weights_npz.py")
    from afetsonar_tf.convert_weights import convert

    return convert(str(EXPORT_DIR), output="")


@pytest.fixture(scope="module")
def tf_outputs(converted_model, golden_teacher_io):
    x_nchw = golden_teacher_io["input"]                  # (2, 6, 256, 256)
    x_nhwc = tf.constant(np.transpose(x_nchw, (0, 2, 3, 1)))
    return converted_model(x_nhwc, training=False)


def _max_diff(tf_tensor, golden_nchw: np.ndarray) -> float:
    got = np.transpose(tf_tensor.numpy(), (0, 3, 1, 2))
    return float(np.max(np.abs(got - golden_nchw)))


def test_damage_logits_parity(tf_outputs, golden_teacher_io):
    damage = tf_outputs["damage_logits"]
    assert len(damage) == 4  # main + 3 aux
    for i, logits in enumerate(damage):
        diff = _max_diff(logits, golden_teacher_io[f"damage_logits_{i}"])
        assert diff <= ATOL, f"damage_logits_{i}: max diff {diff:.2e}"


def test_change_logits_parity(tf_outputs, golden_teacher_io):
    diff = _max_diff(tf_outputs["change_logits"],
                     golden_teacher_io["change_logits"])
    assert diff <= ATOL, f"change_logits: max diff {diff:.2e}"


def test_disaster_logits_parity(tf_outputs, golden_teacher_io):
    got = tf_outputs["disaster_logits"].numpy()
    diff = float(np.max(np.abs(got - golden_teacher_io["disaster_logits"])))
    assert diff <= ATOL, f"disaster_logits: max diff {diff:.2e}"
