"""tf.data pipeline tests on a tiny synthetic TFRecord."""

from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
cv2 = pytest.importorskip("cv2")

from afetsonar_tf.data import (  # noqa: E402
    make_eval_dataset,
    make_train_dataset,
    serialize_example,
)


def _png(arr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", arr)
    assert ok
    return buf.tobytes()


def _write_shard(path, n: int, damaged: bool, hw: int = 96) -> str:
    rng = np.random.default_rng(0 if damaged else 1)
    with tf.io.TFRecordWriter(str(path)) as writer:
        for i in range(n):
            pre = rng.integers(0, 255, (hw, hw, 3), dtype=np.uint8)
            post = rng.integers(0, 255, (hw, hw, 3), dtype=np.uint8)
            mask = np.zeros((hw, hw), dtype=np.uint8)
            anchors = []
            if damaged:
                mask[30:60, 30:60] = 4
                ys, xs = np.nonzero(mask)
                pick = rng.choice(len(ys), size=8, replace=False)
                anchors = [int(v) for pair in zip(ys[pick], xs[pick])
                           for v in pair]
            writer.write(serialize_example(
                pre_png=_png(pre), post_png=_png(post), mask_png=_png(mask),
                height=hw, width=hw, disaster_idx=i % 5,
                anchors=anchors, filename=f"s{i}.png",
            ))
    return str(path)


def test_train_pipeline_shapes_and_ranges(tmp_path):
    dmg = _write_shard(tmp_path / "dmg.tfrecord", 6, damaged=True)
    nodmg = _write_shard(tmp_path / "nodmg.tfrecord", 6, damaged=False)

    ds = make_train_dataset([dmg], [nodmg], global_batch=4, size=64)
    image, mask, disaster = next(iter(ds))

    assert image.shape == (4, 64, 64, 6)
    assert image.dtype == tf.float32
    assert mask.shape == (4, 64, 64)
    assert mask.dtype == tf.int32
    assert disaster.shape == (4,)
    # normalised images: values in a plausible ImageNet-normalised range
    assert float(tf.reduce_min(image)) > -3.5
    assert float(tf.reduce_max(image)) < 3.5
    # masks stay valid classes
    assert int(tf.reduce_min(mask)) >= 0
    assert int(tf.reduce_max(mask)) <= 5


def test_damage_oversampling_biases_batches(tmp_path):
    dmg = _write_shard(tmp_path / "dmg.tfrecord", 4, damaged=True)
    nodmg = _write_shard(tmp_path / "nodmg.tfrecord", 4, damaged=False)

    ds = make_train_dataset([dmg], [nodmg], global_batch=8, size=64,
                            dmg_weight=5.0)
    damaged_fraction = []
    for image, mask, _ in ds.take(12):
        has_damage = tf.reduce_any(mask >= 2, axis=[1, 2])
        damaged_fraction.append(float(tf.reduce_mean(
            tf.cast(has_damage, tf.float32))))
    mean_frac = sum(damaged_fraction) / len(damaged_fraction)
    # weight 5 with equal shard counts -> expect ~5/6 = 0.83 damaged
    assert mean_frac > 0.6, f"oversampling not effective ({mean_frac:.2f})"


def test_eval_pipeline_deterministic(tmp_path):
    shard = _write_shard(tmp_path / "val.tfrecord", 4, damaged=True)
    ds = make_eval_dataset([shard], global_batch=2, size=64)

    first = [(im.numpy(), m.numpy()) for im, m, _ in ds]
    second = [(im.numpy(), m.numpy()) for im, m, _ in ds]
    assert len(first) == 2  # 4 samples / batch 2
    for (im1, m1), (im2, m2) in zip(first, second):
        np.testing.assert_array_equal(im1, im2)
        np.testing.assert_array_equal(m1, m2)
