"""TFRecord schema + tf.data pipelines for TPU training.

Sample layout (written by ``scripts_tf/convert_to_tfrecords.py``):
lossless PNG bytes for pre/post/mask at native resolution, plus up to
16 precomputed damage-pixel crop anchors (replacing the torch
building-aware crop's dynamic pixel sampling).

Shard families reproduce the torch sampling strategy statically:
- ``*_dmg-*``   damaged samples (torch sampler weight 5)
- ``*_nodmg-*`` undamaged samples (weight 1)
- ``*_cp-*``    optional offline Copy-Paste variants (Tier-2)

``make_train_dataset`` mixes them with ``sample_from_datasets``, which
matches ``WeightedRandomSampler(w=5, replacement=True)`` in
expectation while staying fully static-shape for XLA.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import tensorflow as tf

from afetsonar_tf.data.augment_tf import (
    building_aware_crop,
    joint_geometry,
    normalize,
    shared_color_jitter,
)

MAX_ANCHORS = 16

FEATURES = {
    "pre_png": tf.io.FixedLenFeature([], tf.string),
    "post_png": tf.io.FixedLenFeature([], tf.string),
    "mask_png": tf.io.FixedLenFeature([], tf.string),
    "height": tf.io.FixedLenFeature([], tf.int64),
    "width": tf.io.FixedLenFeature([], tf.int64),
    "disaster_idx": tf.io.FixedLenFeature([], tf.int64),
    "n_anchors": tf.io.FixedLenFeature([], tf.int64),
    "anchors": tf.io.FixedLenFeature([MAX_ANCHORS * 2], tf.int64,
                                     default_value=[0] * (MAX_ANCHORS * 2)),
    "filename": tf.io.FixedLenFeature([], tf.string, default_value=b""),
}


def serialize_example(
    pre_png: bytes,
    post_png: bytes,
    mask_png: bytes,
    height: int,
    width: int,
    disaster_idx: int,
    anchors: Sequence[int],
    filename: str = "",
) -> bytes:
    """Build one serialized TFRecord example.

    Args:
        anchors: Flat ``[cy0, cx0, cy1, cx1, ...]`` list of up to
            ``MAX_ANCHORS`` damage-pixel candidates; zero-padded here.
    """
    anchors = list(anchors)[: MAX_ANCHORS * 2]
    n_anchors = len(anchors) // 2
    anchors = anchors + [0] * (MAX_ANCHORS * 2 - len(anchors))

    def _bytes(v: bytes):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))

    def _int(v: int):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[v]))

    feature = {
        "pre_png": _bytes(pre_png),
        "post_png": _bytes(post_png),
        "mask_png": _bytes(mask_png),
        "height": _int(height),
        "width": _int(width),
        "disaster_idx": _int(disaster_idx),
        "n_anchors": _int(n_anchors),
        "anchors": tf.train.Feature(
            int64_list=tf.train.Int64List(value=anchors)),
        "filename": _bytes(filename.encode("utf-8")),
    }
    return tf.train.Example(
        features=tf.train.Features(feature=feature)
    ).SerializeToString()


def _decode(example: dict):
    # decode_image handles both PNG and the optional JPEG-q fallback
    # the converter may use for pre/post; masks are always PNG.
    pre = tf.io.decode_image(example["pre_png"], channels=3,
                             expand_animations=False)
    post = tf.io.decode_image(example["post_png"], channels=3,
                              expand_animations=False)
    mask = tf.io.decode_png(example["mask_png"], channels=1)
    return pre, post, mask


def parse_train(record: tf.Tensor, size: int = 768):
    """Record -> (image (size,size,6) float32, mask (size,size) int32,
    disaster_idx int32) with full training augmentation."""
    ex = tf.io.parse_single_example(record, FEATURES)
    pre, post, mask = _decode(ex)

    pre, post, mask = building_aware_crop(
        pre, post, mask, ex["anchors"], ex["n_anchors"], size=size, prob=0.8,
    )
    pre, post, mask = joint_geometry(pre, post, mask)
    pre, post = shared_color_jitter(pre, post)

    image = tf.concat([normalize(pre), normalize(post)], axis=-1)
    image = tf.ensure_shape(image, [size, size, 6])
    mask = tf.cast(mask[..., 0], tf.int32)
    mask = tf.ensure_shape(mask, [size, size])
    return image, mask, tf.cast(ex["disaster_idx"], tf.int32)


def parse_eval(record: tf.Tensor, size: int = 768):
    """Deterministic eval path: bilinear image / nearest mask resize
    (equivalent of LongestMaxSize on the square xBD tiles)."""
    ex = tf.io.parse_single_example(record, FEATURES)
    pre, post, mask = _decode(ex)

    pre = tf.image.resize(pre, [size, size], method="bilinear")
    post = tf.image.resize(post, [size, size], method="bilinear")
    mask = tf.image.resize(mask, [size, size], method="nearest")

    pre = tf.cast(tf.round(pre), tf.uint8)
    post = tf.cast(tf.round(post), tf.uint8)

    image = tf.concat([normalize(pre), normalize(post)], axis=-1)
    image = tf.ensure_shape(image, [size, size, 6])
    mask = tf.cast(mask[..., 0], tf.int32)
    mask = tf.ensure_shape(mask, [size, size])
    return image, mask, tf.cast(ex["disaster_idx"], tf.int32)


def _shard_dataset(files: Sequence[str]) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(list(files))
    ds = ds.shuffle(max(len(files), 1))
    ds = ds.interleave(
        tf.data.TFRecordDataset,
        cycle_length=min(8, max(len(files), 1)),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    return ds.shuffle(512).repeat()


def make_train_dataset(
    dmg_files: Sequence[str],
    nodmg_files: Sequence[str],
    global_batch: int,
    cp_files: Optional[Sequence[str]] = None,
    size: int = 768,
    dmg_weight: float = 5.0,
) -> tf.data.Dataset:
    """Infinite training dataset with damage oversampling.

    ``dmg_weight`` mirrors ``XBDDatasetV2.get_sample_weights`` (damaged
    rows weighted 5). Use a fixed ``steps_per_epoch`` with this dataset.
    """
    branches: List[tf.data.Dataset] = []
    weights: List[float] = []
    n_dmg, n_nodmg = len(dmg_files), len(nodmg_files)
    if n_dmg:
        branches.append(_shard_dataset(dmg_files))
        weights.append(dmg_weight * n_dmg)
    if n_nodmg:
        branches.append(_shard_dataset(nodmg_files))
        weights.append(1.0 * n_nodmg)
    if not branches:
        raise ValueError("No training shards given")

    total = sum(weights)
    base = tf.data.Dataset.sample_from_datasets(
        branches, weights=[w / total for w in weights]
    ) if len(branches) > 1 else branches[0]

    if cp_files:
        base = tf.data.Dataset.sample_from_datasets(
            [base, _shard_dataset(cp_files)], weights=[0.5, 0.5]
        )

    return (
        base.map(lambda r: parse_train(r, size),
                 num_parallel_calls=tf.data.AUTOTUNE)
        .batch(global_batch, drop_remainder=True)   # mandatory on TPU
        .prefetch(tf.data.AUTOTUNE)
    )


def make_eval_dataset(
    files: Sequence[str],
    global_batch: int,
    size: int = 768,
) -> tf.data.Dataset:
    """Finite, deterministic eval dataset (drop_remainder for TPU —
    pad the shard count or accept dropping < batch leftovers)."""
    ds = tf.data.TFRecordDataset(list(files))
    return (
        ds.map(lambda r: parse_eval(r, size),
               num_parallel_calls=tf.data.AUTOTUNE)
        .batch(global_batch, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
