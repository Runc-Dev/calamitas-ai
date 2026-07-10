"""Convert exported PyTorch weights (.npz) into the Keras teacher.

Runs in the TF environment (transformers 4.x) — consumes the artifacts
produced by ``scripts/export_weights_npz.py`` in the torch env:

- backbone (644 tensors): HF's generic PT->TF loader against the
  ``segformer.encoder.* / decode_head.*``-renamed npz;
- custom heads (56 tensors + 8 skipped ``num_batches_tracked``):
  explicit mapping table with per-tensor shape asserts.

Every npz key must be consumed exactly once; every custom-head Keras
variable assigned exactly once — anything else aborts.

Usage::

    .venv-tf/Scripts/python -m afetsonar_tf.convert_weights \
        --export-dir checkpoints/export \
        --output checkpoints/tf/teacher_v4_ema_tf.ckpt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Set

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf


def _conv_kernel(w: np.ndarray) -> np.ndarray:
    """PT Conv2d (O, I, kH, kW) -> Keras (kH, kW, I, O)."""
    return np.transpose(w, (2, 3, 1, 0))


def load_backbone(model, hf_npz_path: str) -> None:
    """Load encoder + decode head via HF's PT->TF machinery."""
    from transformers.modeling_tf_pytorch_utils import (
        load_pytorch_state_dict_in_tf2_model,
    )

    pt_state = dict(np.load(hf_npz_path))
    _, info = load_pytorch_state_dict_in_tf2_model(
        model.hf, pt_state,
        allow_missing_keys=False,
        output_loading_info=True,
    )
    problems = {k: v for k, v in info.items() if v}
    if problems:
        raise RuntimeError(f"Backbone load not clean: {problems}")
    print(f"Backbone loaded: {len(pt_state)} tensors, no mismatches")


def load_custom_heads(model, full_npz_path: str) -> None:
    """Explicit-table load of fusion/aux/change/disaster head weights."""
    state: Dict[str, np.ndarray] = dict(np.load(full_npz_path))
    custom = {k: v for k, v in state.items()
              if not k.startswith(("encoder.", "decode_head."))}

    consumed: Set[str] = set()

    def take(key: str, expected_shape) -> np.ndarray:
        if key not in custom:
            raise KeyError(f"Expected key missing from npz: {key}")
        arr = custom[key]
        if tuple(arr.shape) != tuple(expected_shape):
            raise ValueError(
                f"{key}: npz shape {arr.shape} != variable "
                f"shape {tuple(expected_shape)}")
        consumed.add(key)
        return arr

    def assign_conv(layer, prefix: str) -> None:
        kernel = _conv_kernel(state[f"{prefix}.weight"])
        consumed.add(f"{prefix}.weight")
        if tuple(kernel.shape) != tuple(layer.kernel.shape):
            raise ValueError(
                f"{prefix}: kernel {kernel.shape} != "
                f"{tuple(layer.kernel.shape)}")
        layer.kernel.assign(kernel)
        if layer.use_bias:
            layer.bias.assign(take(f"{prefix}.bias", layer.bias.shape))

    def assign_bn(layer, prefix: str) -> None:
        layer.gamma.assign(take(f"{prefix}.weight", layer.gamma.shape))
        layer.beta.assign(take(f"{prefix}.bias", layer.beta.shape))
        layer.moving_mean.assign(
            take(f"{prefix}.running_mean", layer.moving_mean.shape))
        layer.moving_variance.assign(
            take(f"{prefix}.running_var", layer.moving_variance.shape))
        consumed.add(f"{prefix}.num_batches_tracked")  # int counter, unused

    def assign_dense(layer, prefix: str) -> None:
        w = take(f"{prefix}.weight",
                 (layer.kernel.shape[1], layer.kernel.shape[0]))
        layer.kernel.assign(np.transpose(w))
        layer.bias.assign(take(f"{prefix}.bias", layer.bias.shape))

    # fusion_convs.{i}: Sequential[Conv2d, BN, ReLU]
    for i, block in enumerate(model.fusion_blocks):
        assign_conv(block.conv, f"fusion_convs.{i}.0")
        assign_bn(block.bn, f"fusion_convs.{i}.1")

    # aux_heads.{i}: Sequential[Conv3x3, BN, ReLU, Dropout2d, Conv1x1]
    if model.aux_heads is not None:
        for i, head in enumerate(model.aux_heads):
            assign_conv(head.conv1, f"aux_heads.{i}.0")
            assign_bn(head.bn, f"aux_heads.{i}.1")
            assign_conv(head.conv2, f"aux_heads.{i}.4")

    # change_head: same Sequential layout
    assign_conv(model.change_head.conv1, "change_head.0")
    assign_bn(model.change_head.bn, "change_head.1")
    assign_conv(model.change_head.conv2, "change_head.4")

    # disaster_head: Sequential[GAP, Flatten, Linear, ReLU, Drop, Linear]
    assign_dense(model.disaster_head.fc1, "disaster_head.2")
    assign_dense(model.disaster_head.fc2, "disaster_head.5")

    leftovers = set(custom) - consumed
    if leftovers:
        raise RuntimeError(
            f"{len(leftovers)} custom-head tensors were never consumed: "
            f"{sorted(leftovers)[:5]}...")
    print(f"Custom heads loaded: {len(consumed)} keys accounted for")


def convert(export_dir: str = "checkpoints/export",
            output: str = "") -> "tf.keras.Model":
    """Build the Keras teacher and load all converted weights.

    Returns the loaded model; optionally saves TF-format weights.
    """
    from afetsonar_tf.models.teacher_tf import build_tf_teacher

    export_path = Path(export_dir)
    model = build_tf_teacher()
    load_backbone(model, str(export_path / "teacher_v4_ema_hf.npz"))
    load_custom_heads(model, str(export_path / "teacher_v4_ema_full.npz"))

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        model.save_weights(output)
        print(f"Saved TF weights -> {output}")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", default="checkpoints/export")
    parser.add_argument("--output",
                        default="checkpoints/tf/teacher_v4_ema_tf.ckpt")
    args = parser.parse_args()
    convert(args.export_dir, args.output)


if __name__ == "__main__":
    main()
