"""Fixtures for the TF-port test suite.

These tests run only where TensorFlow is installed (the .venv-tf
environment or Colab); under the PyTorch env they are skipped wholesale
so the master suite semantics are unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

tf = pytest.importorskip("tensorflow")

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = ROOT / "tests_tf" / "golden"
EXPORT_DIR = ROOT / "checkpoints" / "export"


@pytest.fixture(scope="session")
def golden_loss_data():
    """Small committed fixtures produced by scripts/export_weights_npz.py."""
    import json

    import numpy as np

    inputs_path = GOLDEN_DIR / "loss_inputs.npz"
    values_path = GOLDEN_DIR / "loss_values.json"
    if not inputs_path.exists() or not values_path.exists():
        pytest.skip("golden loss fixtures missing — run "
                    "scripts/export_weights_npz.py in the torch env")
    with open(values_path) as f:
        values = json.load(f)
    return dict(np.load(inputs_path)), values


@pytest.fixture(scope="session")
def golden_teacher_io():
    """Large checkpoint-derived parity fixture (gitignored)."""
    import numpy as np

    path = EXPORT_DIR / "golden_teacher_io.npz"
    if not path.exists():
        pytest.skip("golden_teacher_io.npz missing — run "
                    "scripts/export_weights_npz.py in the torch env")
    return dict(np.load(path))
