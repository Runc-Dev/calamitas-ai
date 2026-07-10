"""TPU-friendly tf.image augmentations.

Mirrors the albumentations training pipeline
(``afetsonar/data/augmentations.py``):

- building-aware crop (80 % centred on a precomputed damage-pixel
  anchor, ±size/4 jitter — anchors come from the TFRecord, replacing
  the torch version's dynamic ``np.where(mask > 0)``)
- RandomRotate90 p=0.5, HFlip p=0.5, VFlip p=0.3 (joint on pre/post/mask)
- mild colour jitter (brightness/contrast ±0.10 p=0.3, hue/sat p=0.2)
  applied with the SAME parameters to pre and post — the equivalent of
  albumentations ``additional_targets={"pre": "image"}``
- ImageNet normalisation

Everything is static-shape and graph-compilable.
"""

from __future__ import annotations

from typing import Tuple

import tensorflow as tf

IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
IMAGENET_STD = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)


def normalize(rgb_uint8: tf.Tensor) -> tf.Tensor:
    """uint8 RGB -> float32 ImageNet-normalised."""
    x = tf.cast(rgb_uint8, tf.float32) / 255.0
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def building_aware_crop(
    pre: tf.Tensor,
    post: tf.Tensor,
    mask: tf.Tensor,
    anchors: tf.Tensor,
    n_anchors: tf.Tensor,
    size: int,
    prob: float = 0.8,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Crop pre/post/mask jointly, preferring damage-anchored windows.

    Args:
        pre: ``(H, W, 3)`` uint8. post: ``(H, W, 3)`` uint8.
        mask: ``(H, W, 1)`` uint8.
        anchors: flat int tensor ``[cy0, cx0, cy1, cx1, ...]`` of
            damage-pixel candidates (may be empty).
        n_anchors: scalar count of valid anchor pairs.
        size: Output side length (source must be >= size).
        prob: Probability of using an anchor-centred crop.

    Returns:
        Cropped ``(size, size, ·)`` pre, post, mask.
    """
    stacked = tf.concat(
        [pre, post, mask], axis=-1
    )                                                    # (H, W, 7) uint8
    h = tf.shape(stacked)[0]
    w = tf.shape(stacked)[1]
    n_anchors = tf.cast(n_anchors, tf.int32)
    anchors = tf.cast(anchors, tf.int32)
    jitter = size // 4

    def _anchor_crop() -> tf.Tensor:
        idx = tf.random.uniform([], 0, tf.maximum(n_anchors, 1), tf.int32)
        cy = anchors[idx * 2]
        cx = anchors[idx * 2 + 1]
        y0 = cy - size // 2 + tf.random.uniform(
            [], -jitter, jitter + 1, tf.int32)
        x0 = cx - size // 2 + tf.random.uniform(
            [], -jitter, jitter + 1, tf.int32)
        y0 = tf.clip_by_value(y0, 0, h - size)
        x0 = tf.clip_by_value(x0, 0, w - size)
        return tf.slice(stacked, [y0, x0, 0], [size, size, 7])

    def _random_crop() -> tf.Tensor:
        return tf.image.random_crop(stacked, [size, size, 7])

    use_anchor = tf.logical_and(
        n_anchors > 0,
        tf.random.uniform([], 0.0, 1.0) < prob,
    )
    out = tf.cond(use_anchor, _anchor_crop, _random_crop)
    out = tf.ensure_shape(out, [size, size, 7])
    return out[..., :3], out[..., 3:6], out[..., 6:7]


def joint_geometry(
    pre: tf.Tensor, post: tf.Tensor, mask: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Rot90 (p=0.5) + horizontal flip (p=0.5) + vertical flip (p=0.3),
    applied identically to all three tensors."""
    stacked = tf.concat([pre, post, mask], axis=-1)

    if tf.random.uniform([]) < 0.5:
        k = tf.random.uniform([], 0, 4, tf.int32)
        stacked = tf.image.rot90(stacked, k)
    if tf.random.uniform([]) < 0.5:
        stacked = tf.image.flip_left_right(stacked)
    if tf.random.uniform([]) < 0.3:
        stacked = tf.image.flip_up_down(stacked)

    return stacked[..., :3], stacked[..., 3:6], stacked[..., 6:7]


def shared_color_jitter(
    pre: tf.Tensor, post: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Mild photometric jitter with identical parameters on both frames.

    Matches RandomBrightnessContrast(±0.10, p=0.3) +
    HueSaturationValue(hue ±5°, sat ±10 %, val ±5 %, p=0.2). Keeping the
    pre/post parameters tied preserves the change-detection signal.
    """
    pre_f = tf.cast(pre, tf.float32) / 255.0
    post_f = tf.cast(post, tf.float32) / 255.0

    # Branch-free probability gating: parameters are scaled towards the
    # identity transform when the draw says "skip". Defining tensors
    # inside `if tf.random...` branches breaks AutoGraph/XLA tracing
    # ("must also be initialized in the else branch").
    do_bc = tf.cast(tf.random.uniform([]) < 0.3, tf.float32)
    brightness = tf.random.uniform([], -0.10, 0.10) * do_bc
    contrast = 1.0 + tf.random.uniform([], -0.10, 0.10) * do_bc

    def _bc(x: tf.Tensor) -> tf.Tensor:
        return tf.clip_by_value(
            (x - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0)

    pre_f, post_f = _bc(pre_f), _bc(post_f)

    do_hsv = tf.cast(tf.random.uniform([]) < 0.2, tf.float32)
    hue = tf.random.uniform([], -5.0 / 360.0, 5.0 / 360.0) * do_hsv
    sat = 1.0 + tf.random.uniform([], -0.10, 0.10) * do_hsv
    val = 1.0 + tf.random.uniform([], -0.05, 0.05) * do_hsv

    def _hsv(x: tf.Tensor) -> tf.Tensor:
        x = tf.image.adjust_hue(x, hue)
        x = tf.image.adjust_saturation(x, sat)
        return tf.clip_by_value(x * val, 0.0, 1.0)

    pre_f, post_f = _hsv(pre_f), _hsv(post_f)

    to_uint8 = lambda x: tf.cast(tf.round(x * 255.0), tf.uint8)
    return to_uint8(pre_f), to_uint8(post_f)
