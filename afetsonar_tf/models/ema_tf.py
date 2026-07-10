"""Manual EMA shadow variables — TF twin of ``afetsonar/models/ema.py``.

Tracks shadows for *trainable* variables only; BatchNorm moving
statistics are intentionally NOT shadowed (identical to the torch
``ModelEMA``, which tracks only ``requires_grad`` parameters).
Create under ``strategy.scope()`` so shadows are mirrored on TPU.
"""

from __future__ import annotations

from typing import Dict, List

import tensorflow as tf


class EmaShadows:
    """Exponential moving average of a model's trainable variables.

    Args:
        model: A built Keras model (variables must already exist).
        decay: EMA decay factor (default 0.999 — matches torch config).

    Example::

        ema = EmaShadows(model, decay=0.999)
        ...
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        ema.update(model)                       # after each step
        ...
        backup = ema.apply_to(model)            # validation with EMA
        evaluate(model)
        ema.restore(model, backup)
    """

    def __init__(self, model: tf.keras.Model, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadows: List[tf.Variable] = [
            tf.Variable(v, trainable=False, name=f"ema/{v.name}")
            for v in model.trainable_variables
        ]

    @tf.function
    def update(self, model: tf.keras.Model) -> None:
        """Shadow <- decay * shadow + (1 - decay) * weight."""
        for shadow, var in zip(self.shadows, model.trainable_variables):
            shadow.assign(self.decay * shadow + (1.0 - self.decay) * var)

    def apply_to(self, model: tf.keras.Model) -> Dict[int, tf.Tensor]:
        """Swap EMA weights into the model; returns a backup for restore."""
        backup = {i: tf.identity(v)
                  for i, v in enumerate(model.trainable_variables)}
        for shadow, var in zip(self.shadows, model.trainable_variables):
            var.assign(shadow)
        return backup

    def restore(self, model: tf.keras.Model,
                backup: Dict[int, tf.Tensor]) -> None:
        """Restore training weights saved by :meth:`apply_to`."""
        variables = model.trainable_variables
        if len(backup) != len(variables):
            raise ValueError(
                f"Backup has {len(backup)} entries but the model has "
                f"{len(variables)} trainable variables"
            )
        for i, var in enumerate(variables):
            var.assign(backup[i])
