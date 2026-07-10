"""Keras twin of ``SiameseTeacherSegformerV3`` (transformers 4.x TF).

Architecture mirrored 1:1 from ``afetsonar/models/teacher.py``:
shared TFSegformer encoder run on pre and post frames, per-stage fusion
``concat(pre, post, |post-pre|) -> 1x1 Conv + BN + ReLU``, HF decode
head, 3 deep-supervision aux heads, binary change head, disaster
classification head.

Tensor-format contract (verified against transformers v4.46 source):
- ``TFSegformerMainLayer`` takes NCHW input and returns hidden_states
  as a 4-tuple of NCHW stage features;
- ``TFSegformerDecodeHead`` consumes NCHW features and returns NHWC
  logits at 1/4 resolution.
Our custom layers all run NHWC (CPU-compatible + TPU-preferred), so we
transpose at the HF boundaries.

Parity-critical details vs torch defaults:
- BatchNormalization(epsilon=1e-5) — Keras default is 1e-3!
- nn.Dropout2d -> SpatialDropout2D (channel-wise), nn.Dropout -> Dropout
- F.interpolate(bilinear, align_corners=False) == tf.image.resize
  bilinear (both use half-pixel centres).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf
from tensorflow.keras import layers  # noqa: E402

#: MiT-B3 hyper-parameters (identical to nvidia/mit-b3) — hardcoded so
#: model construction never needs network access.
B3_CONFIG = dict(
    num_channels=3,
    num_encoder_blocks=4,
    depths=[3, 4, 18, 3],
    sr_ratios=[8, 4, 2, 1],
    hidden_sizes=[64, 128, 320, 512],
    patch_sizes=[7, 3, 3, 3],
    strides=[4, 2, 2, 2],
    num_attention_heads=[1, 2, 5, 8],
    mlp_ratios=[4, 4, 4, 4],
    decoder_hidden_size=768,
)


def build_b3_config(num_labels: int = 6):
    from transformers import SegformerConfig

    return SegformerConfig(num_labels=num_labels, **B3_CONFIG)


class FusionBlock(layers.Layer):
    """concat(pre, post, |diff|) -> 1x1 Conv(no bias) + BN + ReLU (NHWC)."""

    def __init__(self, channels: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(channels, 1, use_bias=False, name="conv")
        self.bn = layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5, name="bn")

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        return tf.nn.relu(self.bn(self.conv(x), training=training))


class AuxHead(layers.Layer):
    """Conv3x3(C->C/2)+BN+ReLU+SpatialDropout(0.1)+Conv1x1(C/2->classes)."""

    def __init__(self, channels: int, num_classes: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(channels // 2, 3, padding="same",
                                   name="conv1")
        self.bn = layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5, name="bn")
        self.drop = layers.SpatialDropout2D(0.1)
        self.conv2 = layers.Conv2D(num_classes, 1, name="conv2")

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = tf.nn.relu(self.bn(self.conv1(x), training=training))
        x = self.drop(x, training=training)
        return self.conv2(x)


class ChangeHead(layers.Layer):
    """Binary change head on the last fused stage (512 -> 256 -> 2)."""

    def __init__(self, in_channels: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(in_channels // 2, 3, padding="same",
                                   name="conv1")
        self.bn = layers.BatchNormalization(
            momentum=0.9, epsilon=1e-5, name="bn")
        self.drop = layers.SpatialDropout2D(0.1)
        self.conv2 = layers.Conv2D(2, 1, name="conv2")

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = tf.nn.relu(self.bn(self.conv1(x), training=training))
        x = self.drop(x, training=training)
        return self.conv2(x)


class DisasterHead(layers.Layer):
    """GAP -> Dense(C/2) + ReLU + Dropout(0.3) -> Dense(num_disaster)."""

    def __init__(self, in_channels: int, num_disaster: int,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.fc1 = layers.Dense(in_channels // 2, name="fc1")
        self.drop = layers.Dropout(0.3)
        self.fc2 = layers.Dense(num_disaster, name="fc2")

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = tf.reduce_mean(x, axis=[1, 2])           # GAP over H, W (NHWC)
        x = tf.nn.relu(self.fc1(x))
        x = self.drop(x, training=training)
        return self.fc2(x)


class TFSiameseTeacherSegformerV3(tf.keras.Model):
    """Siamese SegFormer-B3 teacher — NHWC ``(B, H, W, 6)`` input.

    Output dict matches the torch model (tensor layouts NHWC):
    ``damage_logits`` — list ``[main, aux0, aux1, aux2]`` of
    ``(B, H, W, 6)``; ``change_logits`` — ``(B, H, W, 2)``;
    ``disaster_logits`` — ``(B, 5)``.
    """

    def __init__(
        self,
        num_damage_classes: int = 6,
        num_disaster_classes: int = 5,
        deep_supervision: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        from transformers import TFSegformerForSemanticSegmentation

        config = build_b3_config(num_damage_classes)
        # Keep the full HF model as the holder so HF's PT->TF weight
        # loader can target it directly; we call its sublayers ourselves.
        self.hf = TFSegformerForSemanticSegmentation(config)
        self.segformer = self.hf.segformer
        self.decode_head = self.hf.decode_head

        channels: List[int] = list(config.hidden_sizes)
        self.enc_channels = channels
        self.fusion_blocks = [
            FusionBlock(c, name=f"fusion_{i}")
            for i, c in enumerate(channels)
        ]
        self.aux_heads: Optional[List[AuxHead]] = (
            [AuxHead(c, num_damage_classes, name=f"aux_{i}")
             for i, c in enumerate(channels[:-1])]
            if deep_supervision else None
        )
        self.change_head = ChangeHead(channels[-1], name="change_head")
        self.disaster_head = DisasterHead(
            channels[-1], num_disaster_classes, name="disaster_head")

    # ------------------------------------------------------------------

    def _encode(self, rgb_nhwc: tf.Tensor,
                training: bool) -> List[tf.Tensor]:
        """Shared encoder pass; returns 4 NHWC stage feature maps."""
        x = tf.transpose(rgb_nhwc, [0, 3, 1, 2])      # HF wants NCHW
        out = self.segformer(
            x, output_hidden_states=True, training=training)
        hidden = out.hidden_states                    # 4 x NCHW
        if len(hidden) != 4:
            raise RuntimeError(
                f"Expected 4 encoder stages, got {len(hidden)} — "
                f"transformers TF Segformer layout changed")
        return [tf.transpose(h, [0, 2, 3, 1]) for h in hidden]

    def call(self, x: tf.Tensor, training: bool = False
             ) -> Dict[str, object]:
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]

        feats_pre = self._encode(x[..., :3], training)
        feats_post = self._encode(x[..., 3:6], training)

        fused = [
            self.fusion_blocks[i](
                tf.concat([p, q, tf.abs(q - p)], axis=-1),
                training=training,
            )
            for i, (p, q) in enumerate(zip(feats_pre, feats_post))
        ]

        def up(t: tf.Tensor) -> tf.Tensor:
            return tf.image.resize(
                tf.cast(t, tf.float32), (height, width), method="bilinear")

        # HF decode head consumes NCHW features, emits NHWC logits @1/4.
        fused_nchw = [tf.transpose(f, [0, 3, 1, 2]) for f in fused]
        main_logits = self.decode_head(fused_nchw, training=training)

        damage = [up(main_logits)]
        if self.aux_heads is not None:
            damage += [
                up(self.aux_heads[i](fused[i], training=training))
                for i in range(len(self.aux_heads))
            ]

        return {
            "damage_logits": damage,
            "change_logits": up(self.change_head(fused[-1],
                                                 training=training)),
            "disaster_logits": self.disaster_head(fused[-1],
                                                  training=training),
        }


def build_tf_teacher(
    num_damage_classes: int = 6,
    num_disaster_classes: int = 5,
    deep_supervision: bool = True,
    input_size: int = 128,
) -> TFSiameseTeacherSegformerV3:
    """Construct the teacher and build every variable with a dummy pass."""
    model = TFSiameseTeacherSegformerV3(
        num_damage_classes=num_damage_classes,
        num_disaster_classes=num_disaster_classes,
        deep_supervision=deep_supervision,
    )
    dummy = tf.zeros([1, input_size, input_size, 6], dtype=tf.float32)
    model(dummy, training=False)
    return model
