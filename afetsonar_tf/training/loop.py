"""Custom TPU training loop — TF twin of ``AfetsonarTrainer``'s core.

Semantics mirrored from the torch trainer:
- AdamW(peak_lr, weight_decay=1e-4), linear warmup + cosine decay
  per step (``WarmupCosine``)
- EMA(0.999) updated after every optimiser step; validation runs with
  EMA weights via swap/restore
- global gradient-norm clipping at 1.0 — gradients are all-reduced
  across replicas *before* clipping so the norm matches the torch
  single-device semantics
- ``TeacherLossV3`` multi-task loss; the change mask is derived from
  the damage mask on the fly (classes 2-4, as in
  ``losses/localization.py``)
- bfloat16 mixed precision is set by the caller (notebook); all loss
  math is float32 inside the loss functions themselves

Works under any ``tf.distribute.Strategy`` — the CPU smoke test uses
the default strategy, the notebook uses ``TPUStrategy``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import tensorflow as tf

from afetsonar_tf.losses import teacher_loss_v3_tf
from afetsonar_tf.models.ema_tf import EmaShadows
from afetsonar_tf.training.schedule import WarmupCosine


def derive_change_mask(damage_mask: tf.Tensor) -> tf.Tensor:
    """Binary change mask: damage classes 2/3/4 -> 1, else 0."""
    dm = tf.cast(damage_mask, tf.int32)
    return tf.cast((dm >= 2) & (dm <= 4), tf.int32)


class TeacherTrainerTF:
    """Distributed fine-tuning driver for the TF Siamese teacher.

    Args:
        model: Built Keras teacher (weights already converted/loaded).
        strategy: A ``tf.distribute.Strategy`` (TPUStrategy on Colab).
        total_steps: Planned optimiser steps (epochs * steps_per_epoch).
        peak_lr: Warmup target learning rate (Tier-2 default 1e-5).
        warmup_steps: Requested warmup; clamped by ``WarmupCosine``.
        class_weights: Damage class weights (config.class_weights).
        ema_decay: EMA decay (torch config default 0.999).
    """

    def __init__(
        self,
        model: tf.keras.Model,
        strategy: tf.distribute.Strategy,
        total_steps: int,
        peak_lr: float = 1e-5,
        warmup_steps: Optional[int] = None,
        weight_decay: float = 1e-4,
        class_weights: Optional[Sequence[float]] = None,
        num_damage_classes: int = 6,
        ema_decay: float = 0.999,
    ) -> None:
        self.model = model
        self.strategy = strategy
        self.class_weights = (list(class_weights)
                              if class_weights is not None else None)
        self.num_damage_classes = num_damage_classes

        if warmup_steps is None:
            warmup_steps = max(total_steps // 10, 1)
        with strategy.scope():
            self.schedule = WarmupCosine(peak_lr, total_steps, warmup_steps)
            self.optimizer = tf.keras.optimizers.AdamW(
                learning_rate=self.schedule, weight_decay=weight_decay,
            )
            self.ema = EmaShadows(model, decay=ema_decay)

        self._train_step = tf.function(self._train_step_impl)

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def _loss_fn(self, outputs: Dict[str, Any], mask: tf.Tensor,
                 disaster: tf.Tensor) -> Dict[str, tf.Tensor]:
        targets = {
            "damage_mask": mask,
            "change_mask": derive_change_mask(mask),
            "disaster_idx": disaster,
        }
        return teacher_loss_v3_tf(
            outputs, targets,
            num_damage_classes=self.num_damage_classes,
            damage_class_weights=self.class_weights,
        )

    def _train_step_impl(self, image, mask, disaster) -> tf.Tensor:
        def step_fn(image, mask, disaster):
            n_replicas = self.strategy.num_replicas_in_sync
            with tf.GradientTape() as tape:
                outputs = self.model(image, training=True)
                losses = self._loss_fn(outputs, mask, disaster)
                # Per-replica mean scaled so the cross-replica SUM of
                # gradients equals the global-batch gradient.
                scaled = losses["total"] / n_replicas

            variables = self.model.trainable_variables
            grads = tape.gradient(scaled, variables)

            # All-reduce first, THEN clip: matches torch's global
            # clip_grad_norm_ over the full batch.
            ctx = tf.distribute.get_replica_context()
            if ctx is not None and n_replicas > 1:
                grads = ctx.all_reduce(tf.distribute.ReduceOp.SUM, grads)
            grads, _ = tf.clip_by_global_norm(grads, 1.0)
            self.optimizer.apply_gradients(
                zip(grads, variables),
                experimental_aggregate_gradients=False,
            )
            self.ema.update(self.model)
            return losses["total"]

        per_replica = self.strategy.run(
            step_fn, args=(image, mask, disaster))
        return self.strategy.reduce(
            tf.distribute.ReduceOp.MEAN, per_replica, axis=None)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        train_ds: tf.data.Dataset,
        val_ds: Optional[tf.data.Dataset],
        epochs: int,
        steps_per_epoch: int,
        ckpt_dir: str = "checkpoints/tf",
        log_every: int = 50,
    ) -> List[Dict[str, float]]:
        """Train; validates with EMA weights each epoch; keeps the best.

        Returns the per-epoch history list. Best EMA weights are saved
        to ``<ckpt_dir>/teacher_tf_best_ema.weights.h5``.
        """
        ckpt_path = Path(ckpt_dir)
        ckpt_path.mkdir(parents=True, exist_ok=True)

        dist_train = self.strategy.experimental_distribute_dataset(train_ds)
        train_iter = iter(dist_train)

        history: List[Dict[str, float]] = []
        best_miou = -1.0

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            running = 0.0
            for step in range(steps_per_epoch):
                image, mask, disaster = next(train_iter)
                loss = self._train_step(image, mask, disaster)
                running += float(loss)
                if log_every and (step + 1) % log_every == 0:
                    lr = float(self.schedule(self.optimizer.iterations))
                    print(f"  epoch {epoch} step {step+1}/{steps_per_epoch}"
                          f" loss {running/(step+1):.4f} lr {lr:.2e}",
                          flush=True)

            entry: Dict[str, float] = {
                "epoch": epoch,
                "train_loss": running / max(steps_per_epoch, 1),
                "seconds": time.time() - t0,
            }

            if val_ds is not None:
                val_metrics = self.evaluate(val_ds)
                entry.update({f"val_{k}": v for k, v in val_metrics.items()
                              if isinstance(v, float)})
                if val_metrics["miou_no_bg"] > best_miou:
                    best_miou = val_metrics["miou_no_bg"]
                    backup = self.ema.apply_to(self.model)
                    # TF checkpoint format — h5 is unreliable for
                    # subclassed models (verified by the round-trip test)
                    self.model.save_weights(
                        str(ckpt_path / "teacher_tf_best_ema.ckpt"))
                    self.ema.restore(self.model, backup)
                    entry["is_best"] = 1.0

            history.append(entry)
            with open(ckpt_path / "history.json", "w") as f:
                json.dump(history, f, indent=2)
            print(f"epoch {epoch}: " + ", ".join(
                f"{k}={v:.4f}" for k, v in entry.items() if k != "epoch"),
                flush=True)

        return history

    def evaluate(self, val_ds: tf.data.Dataset) -> Dict[str, Any]:
        """mIoU/mF1 on ``val_ds`` using EMA weights (swap/restore)."""
        from afetsonar.evaluation.metrics import SegmentationMetrics

        backup = self.ema.apply_to(self.model)
        metrics = SegmentationMetrics(num_classes=self.num_damage_classes)

        @tf.function
        def predict_step(image):
            def step_fn(image):
                outputs = self.model(image, training=False)
                logits = outputs["damage_logits"]
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                return tf.argmax(tf.cast(logits, tf.float32), axis=-1)
            return self.strategy.run(step_fn, args=(image,))

        dist_val = self.strategy.experimental_distribute_dataset(val_ds)
        for image, mask, _ in dist_val:
            preds = predict_step(image)
            for p, m in zip(
                self.strategy.experimental_local_results(preds),
                self.strategy.experimental_local_results(mask),
            ):
                metrics.update(p.numpy(), m.numpy())

        self.ema.restore(self.model, backup)
        return metrics.compute()
