"""Convert xBD split CSVs to TFRecord shards for TPU training.

Torch-free (pandas + cv2 + tf.io) — runs in a free Colab CPU session
with Drive mounted, or anywhere the split CSVs and images are visible.

Per split it writes three shard families:
    <split>_dmg-XXXXX.tfrecord     samples containing damage (mask 2-5)
    <split>_nodmg-XXXXX.tfrecord   samples without damage
    <split>_cp-XXXXX.tfrecord      offline Copy-Paste variants of the
                                   damaged samples (train only, optional)

Every sample stores lossless PNG bytes (pre/post/mask) at native
resolution plus up to 16 damage-pixel crop anchors. Missing files ABORT
the conversion (review finding #1 must not be reproduced here); mask
values are asserted to be within 0-5.

Usage (Colab)::

    python scripts_tf/convert_to_tfrecords.py \
        --csv /content/drive/MyDrive/AFETSONAR/data/splits/train_v3.csv \
        --split train \
        --out-dir /content/drive/MyDrive/AFETSONAR/tfrecords \
        --copy-paste

    # repeat with val_v3.csv / test_v3.csv (without --copy-paste);
    # --split gate is the fixed 200-row test subset used by the TPU
    # gate evaluation (notebook 10 globs gate_*.tfrecord)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from afetsonar_tf.data.tfrecords import MAX_ANCHORS, serialize_example  # noqa: E402

SAMPLES_PER_SHARD = 100


def _read_image(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _read_mask(path: str) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    if mask.max() > 5:
        raise ValueError(f"Mask {path} has values > 5 (max {mask.max()}) — "
                         f"the TF losses assume labels 0-5 with no ignore "
                         f"index")
    return mask


def _png_bytes(arr: np.ndarray) -> bytes:
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if arr.ndim == 3 else arr
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes()


def _anchors_from_mask(mask: np.ndarray,
                       rng: np.random.Generator) -> list:
    """Up to MAX_ANCHORS random (cy, cx) damage/building pixels."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return []
    pick = rng.choice(len(ys), size=min(MAX_ANCHORS, len(ys)),
                      replace=False)
    return [int(v) for pair in zip(ys[pick], xs[pick]) for v in pair]


class _ShardWriter:
    """Round-robin sharded TFRecord writer."""

    def __init__(self, out_dir: Path, stem: str) -> None:
        import tensorflow as tf

        self._tf = tf
        self.out_dir = out_dir
        self.stem = stem
        self.count = 0
        self.shard_idx = 0
        self.writer = None

    def write(self, payload: bytes) -> None:
        if self.writer is None or self.count % SAMPLES_PER_SHARD == 0:
            if self.writer is not None:
                self.writer.close()
            path = self.out_dir / f"{self.stem}-{self.shard_idx:05d}.tfrecord"
            self.writer = self._tf.io.TFRecordWriter(str(path))
            self.shard_idx += 1
        self.writer.write(payload)
        self.count += 1

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="split CSV (v3)")
    parser.add_argument("--split", required=True,
                        choices=["train", "val", "test", "gate"])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--copy-paste", action="store_true",
                        help="also write offline Copy-Paste shards "
                             "(train only)")
    parser.add_argument("--jpeg-quality", type=int, default=0,
                        help="if > 0, store pre/post as JPEG at this "
                             "quality instead of PNG (saves ~60%% space; "
                             "mask stays PNG)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from afetsonar.data.copy_paste import CopyPasteAugmentation

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    required = {"post_path", "pre_path", "mask_path", "disaster_idx"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV missing columns: {required - set(df.columns)}")

    def encode_rgb(arr: np.ndarray) -> bytes:
        if args.jpeg_quality > 0:
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(
                ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
            if not ok:
                raise RuntimeError("JPEG encode failed")
            return buf.tobytes()
        return _png_bytes(arr)

    dmg_writer = _ShardWriter(out_dir, f"{args.split}_dmg")
    nodmg_writer = _ShardWriter(out_dir, f"{args.split}_nodmg")
    cp_writer = (_ShardWriter(out_dir, f"{args.split}_cp")
                 if args.copy_paste else None)
    copy_paste = CopyPasteAugmentation(paste_probability=1.0)

    damaged_rows = []
    for _, row in df.iterrows():
        if _read_mask(row["mask_path"]).max() >= 2:
            damaged_rows.append(row)

    n_ok = 0
    for _, row in df.iterrows():
        post = _read_image(row["post_path"])
        pre = _read_image(row["pre_path"])
        mask = _read_mask(row["mask_path"])
        if mask.shape != post.shape[:2]:
            mask = cv2.resize(mask, (post.shape[1], post.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        if pre.shape != post.shape:
            pre = cv2.resize(pre, (post.shape[1], post.shape[0]))

        is_damaged = bool((mask >= 2).any())
        payload = serialize_example(
            pre_png=encode_rgb(pre), post_png=encode_rgb(post),
            mask_png=_png_bytes(mask),
            height=post.shape[0], width=post.shape[1],
            disaster_idx=int(row["disaster_idx"]),
            anchors=_anchors_from_mask(mask, rng),
            filename=str(row.get("filename", "")),
        )
        (dmg_writer if is_damaged else nodmg_writer).write(payload)

        # Offline Copy-Paste variant: damaged donor pasted once.
        if cp_writer is not None and damaged_rows:
            donor = damaged_rows[rng.integers(0, len(damaged_rows))]
            d_post = _read_image(donor["post_path"])
            d_pre = _read_image(donor["pre_path"])
            d_mask = _read_mask(donor["mask_path"])
            out = copy_paste(
                {"post": post, "pre": pre, "mask": mask},
                {"post": d_post, "pre": d_pre, "mask": d_mask},
            )
            cp_writer.write(serialize_example(
                pre_png=encode_rgb(out["pre"]),
                post_png=encode_rgb(out["post"]),
                mask_png=_png_bytes(out["mask"]),
                height=post.shape[0], width=post.shape[1],
                disaster_idx=int(row["disaster_idx"]),
                anchors=_anchors_from_mask(out["mask"], rng),
                filename=str(row.get("filename", "")) + "+cp",
            ))

        n_ok += 1
        if n_ok % 200 == 0:
            print(f"  {n_ok}/{len(df)} samples...", flush=True)

    for w in (dmg_writer, nodmg_writer, cp_writer):
        if w is not None:
            w.close()

    print(f"DONE {args.split}: {n_ok} samples -> "
          f"{dmg_writer.count} damaged / {nodmg_writer.count} undamaged"
          + (f" / {cp_writer.count} copy-paste" if cp_writer else ""))


if __name__ == "__main__":
    main()
