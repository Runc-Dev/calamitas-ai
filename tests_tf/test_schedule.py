"""Tests for the WarmupCosine schedule (incl. review finding #16)."""

from __future__ import annotations

import pytest

tf = pytest.importorskip("tensorflow")

from afetsonar_tf.training.schedule import WarmupCosine  # noqa: E402


def test_warmup_ramps_then_cosine_decays():
    sched = WarmupCosine(peak_lr=1e-3, total_steps=1000, warmup_steps=100)
    assert float(sched(0)) == 0.0
    assert float(sched(50)) == pytest.approx(5e-4, rel=1e-5)
    assert float(sched(100)) == pytest.approx(1e-3, rel=1e-5)
    # cosine midpoint and tail
    assert float(sched(550)) == pytest.approx(5e-4, rel=1e-2)
    assert float(sched(1000)) < 1e-8


def test_warmup_never_swallows_short_runs():
    """Finding #16: requested warmup >= total run must be clamped."""
    sched = WarmupCosine(peak_lr=1e-3, total_steps=100, warmup_steps=100)
    assert sched.warmup_steps == 10  # clamped to 10% of the run
    assert float(sched(99)) < float(sched(10))  # decay actually happens


def test_monotone_decay_after_warmup():
    sched = WarmupCosine(peak_lr=1e-3, total_steps=200, warmup_steps=20)
    values = [float(sched(s)) for s in range(20, 200, 10)]
    assert all(a >= b for a, b in zip(values, values[1:]))
