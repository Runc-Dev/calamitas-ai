"""Ablation study table builder.

Provides utilities to compare model variants across standard metrics and
produce a formatted summary table (as a pandas DataFrame or CSV).

AFETSONAR ablation journey:
  v1 (baseline)  → mIoU_no_bg 0.298
  v2 (+Lovász)   → mIoU_no_bg 0.325   (+0.027)
  v3 (+DeepSup)  → mIoU_no_bg 0.405   (+0.080)  ← biggest single jump
  v4 (+EMA)      → mIoU_no_bg 0.470   (+0.065)
  student (KD)   → mIoU_no_bg 0.395   (12x smaller, 32.8x faster)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


# ============================================================
# Reference results (teacher ablation, xBD test set, 1375 images)
# ============================================================

ABLATION_HISTORY: List[Dict] = [
    {
        "variant": "v1 — baseline (CE only)",
        "miou_no_bg": 0.298,
        "mf1": 0.511,
        "destroyed_iou": 0.412,
        "params_m": 50.3,
        "latency_ms": None,
        "notes": "Baseline SegFormer-B3 Siamese, cross-entropy only",
    },
    {
        "variant": "v2 — +Lovász-Softmax",
        "miou_no_bg": 0.325,
        "mf1": 0.538,
        "destroyed_iou": 0.451,
        "params_m": 50.3,
        "latency_ms": None,
        "notes": "Lovász-Softmax replaces CE; directly optimises mIoU",
    },
    {
        "variant": "v3 — +Lovász +DeepSup",
        "miou_no_bg": 0.405,
        "mf1": 0.612,
        "destroyed_iou": 0.531,
        "params_m": 50.3,
        "latency_ms": None,
        "notes": "Deep supervision (3 aux heads) — biggest single jump (+0.080)",
    },
    {
        "variant": "v4 — +Lovász +DeepSup +EMA (teacher)",
        "miou_no_bg": 0.424,
        "mf1": 0.640,
        "destroyed_iou": 0.570,
        "params_m": 50.3,
        "latency_ms": 1140,
        "notes": "EMA model averaging; final teacher checkpoint",
    },
    {
        "variant": "student — KD (B0)",
        "miou_no_bg": 0.395,
        "mf1": 0.617,
        "destroyed_iou": 0.524,
        "params_m": 4.3,
        "latency_ms": 36,
        "notes": "5-component KD; 12× smaller, 32.8× faster, 93% knowledge retention",
    },
]

SOTA_COMPARISON: List[Dict] = [
    {
        "method": "Durnov 2020 (1st place xView2)",
        "f1": 0.74,
        "params_m": "100+",
        "edge_deployable": False,
        "notes": "Ensemble, GPU-only",
    },
    {
        "method": "Roy et al. 2021",
        "f1": 0.68,
        "params_m": "~80",
        "edge_deployable": False,
        "notes": "EfficientNet backbone",
    },
    {
        "method": "AFETSONAR Teacher (B3)",
        "f1": 0.640,
        "params_m": 50.3,
        "edge_deployable": False,
        "notes": "Single model, no ensemble",
    },
    {
        "method": "AFETSONAR Student (B0) — ours",
        "f1": 0.617,
        "params_m": 4.3,
        "edge_deployable": True,
        "notes": "36 ms latency, Jetson-class hardware",
    },
]


# ============================================================
# Builders
# ============================================================

def build_ablation_table(
    extra_rows: Optional[List[Dict]] = None,
) -> pd.DataFrame:
    """Build the ablation study DataFrame.

    Args:
        extra_rows: Additional experiment rows to append to the default
            AFETSONAR history.

    Returns:
        DataFrame with columns: ``variant``, ``miou_no_bg``, ``mf1``,
        ``destroyed_iou``, ``params_m``, ``latency_ms``, ``notes``.
    """
    rows = list(ABLATION_HISTORY)
    if extra_rows:
        rows.extend(extra_rows)
    df = pd.DataFrame(rows)
    df["delta_miou"] = df["miou_no_bg"].diff().fillna(0.0).round(3)
    return df


def build_sota_table() -> pd.DataFrame:
    """Build the SoTA comparison DataFrame.

    Returns:
        DataFrame with columns: ``method``, ``f1``, ``params_m``,
        ``edge_deployable``, ``notes``.
    """
    return pd.DataFrame(SOTA_COMPARISON)


def save_ablation_results(
    ablation_df: pd.DataFrame,
    sota_df: pd.DataFrame,
    output_dir: str = "results",
) -> None:
    """Persist ablation and SoTA tables to CSV files.

    Args:
        ablation_df: DataFrame from :func:`build_ablation_table`.
        sota_df: DataFrame from :func:`build_sota_table`.
        output_dir: Directory where CSV files will be written.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ablation_path = Path(output_dir) / "ablation_table.csv"
    sota_path = Path(output_dir) / "sota_comparison.csv"
    ablation_df.to_csv(ablation_path, index=False)
    sota_df.to_csv(sota_path, index=False)
    print(f"Saved → {ablation_path}")
    print(f"Saved → {sota_path}")
