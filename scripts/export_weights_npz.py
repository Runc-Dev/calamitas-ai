"""Export teacher weights + golden fixtures for the TensorFlow port.

Runs in the PYTORCH environment (transformers 5.x). Produces
framework-neutral artifacts consumed by the TF environment
(transformers 4.x) — the two environments never coexist.

Outputs (checkpoints/export/, gitignored — regenerate on demand):
    teacher_v4_ema_full.npz   all state-dict tensors, original names
    teacher_v4_ema_hf.npz     encoder/decode_head subset renamed to plain
                              SegformerForSemanticSegmentation layout
                              (encoder.X -> segformer.encoder.X) so HF's
                              generic PTtoTF loader can consume it
    golden_teacher_io.npz     fixed-seed input (2,6,256,256) + fp32 CPU
                              eval outputs (4 damage logits, change,
                              disaster) for the TF parity test
    manifest.json             key->shape map + source metadata

Outputs (tests_tf/golden/, small, committed to git):
    loss_inputs.npz           fixed-seed logits/targets for loss tests
    loss_values.json          TeacherLossV3 + component golden values

Usage::

    python scripts/export_weights_npz.py \
        --checkpoint checkpoints/teacher_v4_best_ema.pth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BACKBONE_PREFIXES = ("encoder.", "decode_head.")


def load_effective_state(checkpoint_path: str) -> dict:
    """Return the inference-effective state dict.

    teacher_v4_best_ema.pth stores EMA-applied weights directly in
    ``model_state_dict``. Newer trainer checkpoints keep a separate
    ``ema_state`` holding only parameters — overlay it on top of the
    full state so BN running stats survive.
    """
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError("Unexpected checkpoint format (not a dict)")

    state = None
    for key in ("model_state_dict", "state_dict", "model"):
        if key in ckpt:
            state = dict(ckpt[key])
            break
    if state is None:
        state = {k: v for k, v in ckpt.items() if hasattr(v, "numpy")}

    if "ema_state" in ckpt:  # parameter-only shadow -> overlay
        state.update(ckpt["ema_state"])

    meta = {
        "source": str(checkpoint_path),
        "epoch": int(ckpt.get("epoch", -1)),
        "val_miou_no_bg": float(ckpt.get("val_miou_no_bg", float("nan"))),
        "n_keys": len(state),
    }
    return state, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",
                        default="checkpoints/teacher_v4_best_ema.pth")
    parser.add_argument("--out-dir", default="checkpoints/export")
    parser.add_argument("--golden-dir", default="tests_tf/golden")
    args = parser.parse_args()

    import torch

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    golden_dir = Path(args.golden_dir)
    golden_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. State dict -> npz ------------------------------------------
    state, meta = load_effective_state(args.checkpoint)
    arrays = {k: v.detach().cpu().numpy().astype(np.float32)
              for k, v in state.items()}

    np.savez(out_dir / "teacher_v4_ema_full.npz", **arrays)

    hf_arrays = {}
    for k, v in arrays.items():
        if k.startswith("encoder."):
            hf_arrays["segformer." + k] = v
        elif k.startswith("decode_head."):
            hf_arrays[k] = v
    np.savez(out_dir / "teacher_v4_ema_hf.npz", **hf_arrays)

    n_custom = len(arrays) - len(hf_arrays)
    print(f"Exported {len(arrays)} tensors "
          f"({len(hf_arrays)} backbone, {n_custom} custom-head)")

    # ---- 2. Golden I/O for the parity test ----------------------------
    from afetsonar.models.teacher import SiameseTeacherSegformerV3

    model = SiameseTeacherSegformerV3(pretrained=False)
    result = model.load_state_dict(state, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    model.eval()

    torch.manual_seed(1234)
    x = torch.randn(2, 6, 256, 256, dtype=torch.float32)
    with torch.no_grad():
        out = model(x)

    dmg = out["damage_logits"]
    dmg = dmg if isinstance(dmg, (list, tuple)) else [dmg]
    golden_io = {"input": x.numpy()}
    for i, t in enumerate(dmg):
        golden_io[f"damage_logits_{i}"] = t.numpy()
    golden_io["change_logits"] = out["change_logits"].numpy()
    golden_io["disaster_logits"] = out["disaster_logits"].numpy()
    np.savez(out_dir / "golden_teacher_io.npz", **golden_io)
    print("Golden IO:",
          {k: tuple(v.shape) for k, v in golden_io.items()})

    # ---- 3. Golden loss fixtures (small, committed) --------------------
    from afetsonar.config import DefaultConfig
    from afetsonar.losses.combo import (
        ComboDamageLossV3, DiceLoss, FocalLoss, TeacherLossV3,
    )
    from afetsonar.losses.lovasz import LovaszSoftmaxLoss

    cfg = DefaultConfig()
    g = torch.Generator().manual_seed(4321)
    logits_main = torch.randn(2, 6, 64, 64, generator=g)
    logits_aux = [torch.randn(2, 6, 64, 64, generator=g) for _ in range(3)]
    change_logits = torch.randn(2, 2, 64, 64, generator=g)
    disaster_logits = torch.randn(2, 5, generator=g)
    damage_mask = torch.randint(0, 6, (2, 64, 64), generator=g)
    change_mask = ((damage_mask >= 2) & (damage_mask <= 4)).long()
    disaster_idx = torch.randint(0, 5, (2,), generator=g)

    with torch.no_grad():
        teacher_loss = TeacherLossV3(
            damage_class_weights=cfg.class_weights,
        )({"damage_logits": [logits_main, *logits_aux],
           "change_logits": change_logits,
           "disaster_logits": disaster_logits},
          {"damage_mask": damage_mask,
           "change_mask": change_mask,
           "disaster_idx": disaster_idx})

        combo = ComboDamageLossV3(
            num_classes=6, class_weights=cfg.class_weights,
        )(logits_main, damage_mask)
        lovasz = LovaszSoftmaxLoss(classes="present")(logits_main, damage_mask)
        lovasz_w = LovaszSoftmaxLoss(
            classes="present", class_weights=cfg.class_weights,
        )(logits_main, damage_mask)
        dice = DiceLoss(
            num_classes=6, class_weights=cfg.class_weights,
        )(logits_main, damage_mask)
        focal = FocalLoss(
            gamma=2.0, alpha=cfg.class_weights,
        )(logits_main, damage_mask)

    np.savez(
        golden_dir / "loss_inputs.npz",
        logits_main=logits_main.numpy(),
        logits_aux_0=logits_aux[0].numpy(),
        logits_aux_1=logits_aux[1].numpy(),
        logits_aux_2=logits_aux[2].numpy(),
        change_logits=change_logits.numpy(),
        disaster_logits=disaster_logits.numpy(),
        damage_mask=damage_mask.numpy().astype(np.int32),
        change_mask=change_mask.numpy().astype(np.int32),
        disaster_idx=disaster_idx.numpy().astype(np.int32),
        class_weights=np.asarray(cfg.class_weights, dtype=np.float32),
    )
    loss_values = {
        "teacher_total": float(teacher_loss["total"]),
        "teacher_change": float(teacher_loss["change"]),
        "teacher_disaster": float(teacher_loss["disaster"]),
        "teacher_damage": float(teacher_loss["damage"]),
        "combo_total": float(combo["total"]),
        "combo_lovasz": float(combo["lovasz"]),
        "combo_dice": float(combo["dice"]),
        "combo_focal": float(combo["focal"]),
        "lovasz_unweighted": float(lovasz),
        "lovasz_weighted": float(lovasz_w),
        "dice": float(dice),
        "focal": float(focal),
    }
    with open(golden_dir / "loss_values.json", "w") as f:
        json.dump(loss_values, f, indent=2)
    print("Golden losses:", json.dumps(loss_values, indent=2))

    # ---- 4. Manifest ----------------------------------------------------
    import hashlib
    import subprocess

    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        git_commit = "unknown"

    manifest = {
        **meta,
        "git_commit": git_commit,
        "golden_input_seed": 1234,
        "golden_loss_seed": 4321,
        "model_config": {
            "architecture": "SiameseTeacherSegformerV3",
            "backbone": "MiT-B3",
            "num_damage_classes": 6,
            "num_disaster_classes": 5,
            "deep_supervision": True,
        },
        "backbone_keys": len(hf_arrays),
        "custom_keys": sorted(
            k for k in arrays if not k.startswith(BACKBONE_PREFIXES)
        ),
        "shapes": {k: list(v.shape) for k, v in arrays.items()},
        "dtypes": {k: str(v.dtype) for k, v in arrays.items()},
        "sha256": {
            name: _sha256(out_dir / name)
            for name in ("teacher_v4_ema_full.npz",
                         "teacher_v4_ema_hf.npz",
                         "golden_teacher_io.npz")
        },
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written -> {out_dir/'manifest.json'} "
          f"(sha256 + git {git_commit[:8]})")


if __name__ == "__main__":
    main()
