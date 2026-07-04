"""Version-tolerant access to HuggingFace SegFormer internals.

The AFETSONAR models keep only the encoder and decode head of
``SegformerForSemanticSegmentation``.  The attribute path to those
submodules (``base.segformer.encoder`` etc.) has changed between
transformers releases, so we resolve them defensively: try the known
attribute paths first, then fall back to searching the module tree by
class name, and fail loudly (never silently) if nothing matches.

Checkpoints in this project were validated against transformers 5.7.0
(708/708 state-dict keys match).
"""

from __future__ import annotations

import torch.nn as nn


def _fail(model: nn.Module, wanted: str) -> RuntimeError:
    import transformers

    return RuntimeError(
        f"Cannot locate {wanted} inside {type(model).__name__} "
        f"(transformers {transformers.__version__}). The HuggingFace "
        f"module layout changed in this release. Install a tested "
        f"version instead: pip install 'transformers==5.7.0'"
    )


def get_segformer_encoder(model: nn.Module) -> nn.Module:
    """Return the ``SegformerEncoder`` submodule of a HF SegFormer model.

    Args:
        model: A ``SegformerForSemanticSegmentation`` (or compatible)
            instance from any transformers version.

    Returns:
        The encoder module (hierarchical Mix Transformer).

    Raises:
        RuntimeError: If no encoder can be found — with instructions to
            install the validated transformers version.
    """
    seg = getattr(model, "segformer", model)
    encoder = getattr(seg, "encoder", None)
    if encoder is not None:
        return encoder
    for module in model.modules():
        if type(module).__name__ == "SegformerEncoder":
            return module
    raise _fail(model, "SegformerEncoder")


def get_segformer_decode_head(model: nn.Module) -> nn.Module:
    """Return the ``SegformerDecodeHead`` submodule of a HF SegFormer model.

    Args:
        model: A ``SegformerForSemanticSegmentation`` (or compatible)
            instance from any transformers version.

    Returns:
        The all-MLP decode head.

    Raises:
        RuntimeError: If no decode head can be found.
    """
    head = getattr(model, "decode_head", None)
    if head is not None:
        return head
    for module in model.modules():
        if type(module).__name__ == "SegformerDecodeHead":
            return module
    raise _fail(model, "SegformerDecodeHead")
