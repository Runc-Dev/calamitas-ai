"""Pytest fixtures shared across all AFETSONAR test modules."""

from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.fixture
def device() -> torch.device:
    """Return CPU device for deterministic unit tests."""
    return torch.device("cpu")


@pytest.fixture
def dummy_siamese_batch(device: torch.device):
    """Return a small 6-channel pre+post batch on CPU."""
    torch.manual_seed(42)
    return torch.randn(2, 6, 64, 64, device=device)


@pytest.fixture
def dummy_rgb_batch(device: torch.device):
    """Return a small 3-channel RGB batch on CPU."""
    torch.manual_seed(42)
    return torch.randn(2, 3, 64, 64, device=device)


@pytest.fixture
def dummy_mask(device: torch.device):
    """Return a small (2, 64, 64) label mask with values 0–5."""
    torch.manual_seed(42)
    return torch.randint(0, 6, (2, 64, 64), device=device)


@pytest.fixture
def dummy_binary_mask(device: torch.device):
    """Return a small (2, 64, 64) binary label mask with values 0–1."""
    torch.manual_seed(42)
    return torch.randint(0, 2, (2, 64, 64), device=device)


@pytest.fixture
def dummy_buildings():
    """Return a small list of building dicts for routing tests."""
    return [
        {"building_id": i, "lat": 41.005 + i * 0.001, "lon": 28.977 + i * 0.001,
         "damage_class": (i % 5) + 1,
         "damage_class_name": ["no_damage", "minor_damage", "major_damage", "destroyed", "unclassified"][i % 5],
         "area_m2": 100.0 + i * 10.0,
         "priority_score": float(i * 2),
         "survival_prob": 0.8 - i * 0.05}
        for i in range(10)
    ]
