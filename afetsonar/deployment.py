"""ONNX export utilities for edge deployment (Tier 4 roadmap).

Exports the Siamese damage-classification models to ONNX so they can be
served by onnxruntime or converted to a TensorRT engine on an edge
device (Jetson Nano / Xavier on the drone).

The exported graph takes a single ``(N, 6, H, W)`` float tensor — the
channel-concatenated pre+post image pair, preprocessed exactly like
:meth:`afetsonar.pipeline.AfetsonarPipeline.predict` — and returns the
``(N, 6, H, W)`` damage logits.

Example:
    >>> from afetsonar.deployment import export_to_onnx, verify_onnx
    >>> path = export_to_onnx(
    ...     "checkpoints/student/student_v1_best_ema.pth",
    ...     "checkpoints/student_v1.onnx",
    ... )
    >>> verify_onnx("checkpoints/student/student_v1_best_ema.pth", path)

References:
    ONNX: Bai et al. 2019 — Open Neural Network Exchange.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import torch

from afetsonar.pipeline import AfetsonarPipeline

PathLike = Union[str, Path]


class _LogitsOnly(torch.nn.Module):
    """Wrap a Siamese model so ONNX sees a single tensor output.

    The models return a dict (damage/change/disaster logits plus KD
    features); ONNX export needs flat tensor outputs, and edge
    inference only ever consumes the main damage logits.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pre_post: torch.Tensor) -> torch.Tensor:
        out = self.model(pre_post)
        logits = out["damage_logits"]
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        return logits


def _load_wrapper(checkpoint_path: PathLike) -> _LogitsOnly:
    """Load a checkpoint (student or teacher, auto-detected) on CPU."""
    pipeline = AfetsonarPipeline(str(checkpoint_path), device="cpu")
    return _LogitsOnly(pipeline.model).eval()


def export_to_onnx(
    checkpoint_path: PathLike,
    output_path: PathLike,
    image_size: int = 512,
    opset: int = 17,
) -> str:
    """Export a checkpoint to ONNX with a dynamic batch axis.

    Args:
        checkpoint_path: ``.pth`` checkpoint (student or teacher —
            architecture is auto-detected by ``AfetsonarPipeline``).
        output_path: Destination ``.onnx`` file.
        image_size: Spatial size the graph is traced at.  Must match
            the size used at inference time (default 512, the training
            resolution).
        opset: ONNX opset version.

    Returns:
        The output path as ``str``.
    """
    wrapper = _load_wrapper(checkpoint_path)
    dummy = torch.randn(1, 6, image_size, image_size)

    kwargs = dict(
        input_names=["pre_post"],
        output_names=["damage_logits"],
        dynamic_axes={"pre_post": {0: "batch"}, "damage_logits": {0: "batch"}},
        opset_version=opset,
    )
    with torch.no_grad():
        try:
            torch.onnx.export(wrapper, (dummy,), str(output_path), dynamo=False, **kwargs)
        except TypeError:
            # Older torch without the ``dynamo`` flag.
            torch.onnx.export(wrapper, (dummy,), str(output_path), **kwargs)

    try:
        import onnx

        onnx.checker.check_model(onnx.load(str(output_path)))
    except ImportError:
        pass  # structural check is optional — onnxruntime still validates on load

    return str(output_path)


def verify_onnx(
    checkpoint_path: PathLike,
    onnx_path: PathLike,
    image_size: int = 512,
    atol: float = 1e-3,
) -> float:
    """Check onnxruntime output parity against the torch model.

    Args:
        checkpoint_path: The ``.pth`` checkpoint the graph was exported from.
        onnx_path: The exported ``.onnx`` file.
        image_size: Spatial size to test at (must match export).
        atol: Maximum tolerated absolute difference.

    Returns:
        The maximum absolute difference between torch and onnxruntime
        logits.

    Raises:
        AssertionError: If the difference exceeds ``atol``.
    """
    import onnxruntime as ort

    wrapper = _load_wrapper(checkpoint_path)
    torch.manual_seed(0)
    dummy = torch.randn(1, 6, image_size, image_size)

    with torch.no_grad():
        torch_out = wrapper(dummy).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    (ort_out,) = session.run(None, {"pre_post": dummy.numpy()})

    max_diff = float(np.max(np.abs(torch_out - ort_out)))
    if max_diff > atol:
        raise AssertionError(
            f"ONNX/torch mismatch: max |diff| = {max_diff:.2e} > atol {atol:.0e}"
        )
    return max_diff
