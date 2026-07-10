"""Lovász-Softmax loss in TensorFlow — XLA/TPU-friendly static form.

Port of ``afetsonar/losses/lovasz.py`` (Berman et al. 2018, CVPR).
The PyTorch version dynamically skips classes absent from the batch
("present" mode) — dynamic shapes break XLA compilation, so this port
computes every class unconditionally and masks absent classes out of
the reduction with presence indicators. Mathematically identical to
the torch "present" mean (verified by golden-value tests).

Note: ``ignore_index`` filtering is intentionally NOT implemented —
xBD masks contain only values 0–5 (the TFRecord converter asserts
this). All math runs in float32 regardless of the compute policy.
"""

from __future__ import annotations

from typing import Optional, Sequence

import tensorflow as tf


def lovasz_softmax_tf(
    logits: tf.Tensor,
    labels: tf.Tensor,
    class_weights: Optional[Sequence[float]] = None,
) -> tf.Tensor:
    """Multi-class Lovász-Softmax loss over a flattened batch.

    Args:
        logits: ``(B, H, W, C)`` raw logits (NHWC), any float dtype.
        labels: ``(B, H, W)`` integer class labels in ``[0, C)``.
        class_weights: Optional per-class weights (length ``C``).
            Matches the torch behaviour: weighted mean over classes
            present in the batch.

    Returns:
        Scalar float32 loss.
    """
    num_classes = logits.shape[-1]
    if num_classes is None:
        raise ValueError("Channel dimension must be static")

    probas = tf.nn.softmax(tf.cast(logits, tf.float32), axis=-1)
    probas = tf.reshape(probas, [-1, num_classes])       # (N, C)
    labels_flat = tf.reshape(tf.cast(labels, tf.int32), [-1])  # (N,)
    n = tf.shape(probas)[0]

    per_class_losses = []
    presences = []
    for c in range(num_classes):
        fg = tf.cast(tf.equal(labels_flat, c), tf.float32)     # (N,)
        errors = tf.abs(fg - probas[:, c])
        errors_sorted, perm = tf.math.top_k(errors, k=n, sorted=True)
        fg_sorted = tf.gather(fg, perm)

        gts = tf.reduce_sum(fg)
        intersection = gts - tf.cumsum(fg_sorted)
        union = gts + tf.cumsum(1.0 - fg_sorted)
        jaccard = 1.0 - intersection / tf.maximum(union, 1e-12)
        # Lovász gradient: first element kept, rest differenced.
        jaccard = tf.concat([jaccard[:1], jaccard[1:] - jaccard[:-1]], axis=0)

        per_class_losses.append(tf.tensordot(errors_sorted, jaccard, 1))
        presences.append(tf.cast(gts > 0, tf.float32))

    losses = tf.stack(per_class_losses)                  # (C,)
    present = tf.stack(presences)                        # (C,)
    if class_weights is not None:
        weights = tf.constant(list(class_weights), dtype=tf.float32)
    else:
        weights = tf.ones([num_classes], dtype=tf.float32)

    denom = tf.reduce_sum(weights * present)
    return tf.reduce_sum(losses * weights * present) / tf.maximum(denom, 1e-12)
