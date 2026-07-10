"""Linear-warmup + cosine-decay learning-rate schedule.

Mirrors the PyTorch trainer's per-batch LambdaLR (linear ramp over
``min(2, epochs)`` epochs, cosine to zero afterwards) and fixes review
finding #16: warmup can never swallow the whole run — it is capped at
10 % of total steps.
"""

from __future__ import annotations

import math

import tensorflow as tf


class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Per-step linear warmup followed by cosine decay to zero.

    Args:
        peak_lr: Learning rate at the end of warmup.
        total_steps: Total optimiser steps in the run.
        warmup_steps: Requested warmup length; clamped to
            ``max(1, min(warmup_steps, total_steps // 10))`` so short
            runs still train (finding #16).
    """

    def __init__(self, peak_lr: float, total_steps: int,
                 warmup_steps: int) -> None:
        super().__init__()
        if total_steps < 1:
            raise ValueError("total_steps must be >= 1")
        self.peak_lr = float(peak_lr)
        self.total_steps = int(total_steps)
        self.warmup_steps = max(1, min(int(warmup_steps),
                                       max(self.total_steps // 10, 1)))

    def __call__(self, step) -> tf.Tensor:
        step_f = tf.cast(step, tf.float32)
        warmup = tf.constant(float(self.warmup_steps), tf.float32)
        total = tf.constant(float(self.total_steps), tf.float32)

        warm_lr = self.peak_lr * step_f / warmup
        progress = (step_f - warmup) / tf.maximum(total - warmup, 1.0)
        progress = tf.clip_by_value(progress, 0.0, 1.0)
        cos_lr = self.peak_lr * 0.5 * (1.0 + tf.cos(math.pi * progress))
        return tf.where(step_f < warmup, warm_lr, cos_lr)

    def get_config(self) -> dict:
        return {"peak_lr": self.peak_lr, "total_steps": self.total_steps,
                "warmup_steps": self.warmup_steps}
