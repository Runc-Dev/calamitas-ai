"""CPU smoke test for the training loop (schedule + EMA + loss wiring).

Uses a tiny stand-in model with the same output contract as the real
teacher so 20 optimisation steps finish in seconds — the full model is
exercised on TPU by notebook 10.
"""

from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
cv2 = pytest.importorskip("cv2")

from afetsonar_tf.data import make_train_dataset, serialize_example  # noqa: E402
from afetsonar_tf.training.loop import TeacherTrainerTF, derive_change_mask  # noqa: E402


class _ToyTeacher(tf.keras.Model):
    """Minimal model with the teacher's output contract (NHWC)."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = tf.keras.layers.Conv2D(16, 3, padding="same",
                                               activation="relu")
        self.damage = tf.keras.layers.Conv2D(6, 1)
        self.change = tf.keras.layers.Conv2D(2, 1)
        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.disaster = tf.keras.layers.Dense(5)

    def call(self, x, training=False):
        h = self.backbone(x)
        return {
            "damage_logits": [self.damage(h)],
            "change_logits": self.change(h),
            "disaster_logits": self.disaster(self.pool(h)),
        }


def _write_shard(tmp_path, n=8, hw=64):
    rng = np.random.default_rng(0)
    path = tmp_path / "train.tfrecord"
    with tf.io.TFRecordWriter(str(path)) as w:
        for i in range(n):
            pre = rng.integers(0, 255, (hw, hw, 3), dtype=np.uint8)
            post = rng.integers(0, 255, (hw, hw, 3), dtype=np.uint8)
            mask = np.zeros((hw, hw), dtype=np.uint8)
            mask[10:30, 10:30] = rng.integers(1, 6)
            ys, xs = np.nonzero(mask)
            pick = rng.choice(len(ys), size=8, replace=False)
            anchors = [int(v) for p_ in zip(ys[pick], xs[pick]) for v in p_]

            def png(a):
                ok, buf = cv2.imencode(".png", a)
                assert ok
                return buf.tobytes()

            w.write(serialize_example(
                pre_png=png(pre), post_png=png(post), mask_png=png(mask),
                height=hw, width=hw, disaster_idx=i % 5, anchors=anchors,
            ))
    return str(path)


def test_derive_change_mask():
    mask = tf.constant([[0, 1, 2], [3, 4, 5]])
    change = derive_change_mask(mask).numpy()
    np.testing.assert_array_equal(change, [[0, 0, 1], [1, 1, 0]])


def test_twenty_training_steps_no_nan(tmp_path):
    shard = _write_shard(tmp_path)
    ds = make_train_dataset([shard], [], global_batch=2, size=32)

    strategy = tf.distribute.get_strategy()  # default (CPU)
    model = _ToyTeacher()
    model(tf.zeros([1, 32, 32, 6]))  # build

    trainer = TeacherTrainerTF(
        model, strategy, total_steps=20, peak_lr=1e-3,
        class_weights=[0.5, 1.0, 5.0, 3.0, 2.5, 2.0],
    )

    dist = strategy.experimental_distribute_dataset(ds)
    it = iter(dist)
    losses = []
    for _ in range(20):
        image, mask, disaster = next(it)
        losses.append(float(trainer._train_step(image, mask, disaster)))

    assert all(np.isfinite(losses)), f"non-finite loss: {losses}"
    # optimisation should make some progress on 8 memorisable samples
    assert min(losses[10:]) < losses[0]


def test_evaluate_returns_metrics(tmp_path):
    shard = _write_shard(tmp_path, n=4)
    from afetsonar_tf.data import make_eval_dataset

    ds = make_eval_dataset([shard], global_batch=2, size=32)
    strategy = tf.distribute.get_strategy()
    model = _ToyTeacher()
    model(tf.zeros([1, 32, 32, 6]))

    trainer = TeacherTrainerTF(model, strategy, total_steps=10)
    metrics = trainer.evaluate(ds)

    for key in ("miou", "miou_no_bg", "mf1", "accuracy"):
        assert key in metrics
        assert np.isfinite(metrics[key])
