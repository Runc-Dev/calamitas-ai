"""Export an AFETSONAR checkpoint to ONNX for edge deployment.

Usage::

    python scripts/export_onnx.py \
        --checkpoint checkpoints/student/student_v1_best_ema.pth \
        --output checkpoints/student_v1.onnx

Requires: pip install onnx onnxruntime
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without installing the package
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export an AFETSONAR model (student or teacher) to ONNX",
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to the .pth checkpoint (architecture auto-detected)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output .onnx path (default: checkpoint path with .onnx suffix)",
    )
    parser.add_argument(
        "--image-size", type=int, default=512,
        help="Spatial size to trace at — must match inference size (default: 512)",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument(
        "--no-verify", action="store_true",
        help="Skip the onnxruntime parity check",
    )
    parser.add_argument(
        "--atol", type=float, default=1e-3,
        help="Parity check tolerance (default: 1e-3)",
    )
    args = parser.parse_args()

    from afetsonar.deployment import export_to_onnx, verify_onnx

    output = args.output or str(Path(args.checkpoint).with_suffix(".onnx"))
    path = export_to_onnx(
        args.checkpoint, output, image_size=args.image_size, opset=args.opset,
    )
    size_mb = Path(path).stat().st_size / 1e6
    print(f"Exported: {path} ({size_mb:.1f} MB)")

    if not args.no_verify:
        max_diff = verify_onnx(
            args.checkpoint, path, image_size=args.image_size, atol=args.atol,
        )
        print(f"Parity OK — max |torch - onnxruntime| = {max_diff:.2e} (atol {args.atol:.0e})")


if __name__ == "__main__":
    main()
