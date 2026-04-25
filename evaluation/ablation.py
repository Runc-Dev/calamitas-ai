"""Helpers for building and plotting the AFETSONAR ablation study table.

The ablation records every incremental architectural or training change
and its effect on ``mIoU_no_bg``. The hand-entered history from the notebook
is mirrored here so that :mod:`scripts.evaluate` can produce the canonical
figure / CSV without needing the notebook runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Union

import pandas as pd


@dataclass(frozen=True)
class AblationRow:
    """One row of the ablation table."""

    stage: str
    configuration: str
    miou_no_bg: float
    mf1: float
    delta: float  # Change vs. previous stage.
    notes: str = ""


# The canonical ablation history captured at the end of Phase 2.
DEFAULT_ABLATION: List[AblationRow] = [
    AblationRow(
        stage="v1",
        configuration="SegFormer-B3 + BCE",
        miou_no_bg=0.298,
        mf1=0.502,
        delta=0.0,
        notes="Vanilla baseline.",
    ),
    AblationRow(
        stage="v2",
        configuration="+ Focal + class weights",
        miou_no_bg=0.325,
        mf1=0.528,
        delta=0.027,
        notes="Class-balanced focal loss.",
    ),
    AblationRow(
        stage="v3",
        configuration="+ Combo (Lovász + Dice + Focal)",
        miou_no_bg=0.405,
        mf1=0.598,
        delta=0.080,
        notes="Lovász term directly optimizes mIoU.",
    ),
    AblationRow(
        stage="v4",
        configuration="+ Deep supervision + EMA + warm restarts",
        miou_no_bg=0.470,
        mf1=0.650,
        delta=0.065,
        notes="Best teacher checkpoint (teacher_v4_best_ema.pth).",
    ),
    AblationRow(
        stage="student",
        configuration="SegFormer-B0 + 5-comp KD from v4",
        miou_no_bg=0.395,
        mf1=0.617,
        delta=-0.075,
        notes="12× smaller, edge-deployable, ~93% knowledge retention.",
    ),
]


def ablation_to_dataframe(
    rows: Optional[Iterable[AblationRow]] = None,
) -> pd.DataFrame:
    """Return the ablation table as a ``pandas.DataFrame``."""
    rows = list(rows) if rows is not None else DEFAULT_ABLATION
    return pd.DataFrame(
        [
            {
                "stage": r.stage,
                "configuration": r.configuration,
                "miou_no_bg": r.miou_no_bg,
                "mf1": r.mf1,
                "delta": r.delta,
                "notes": r.notes,
            }
            for r in rows
        ]
    )


def write_ablation_csv(
    output_path: Union[str, Path],
    rows: Optional[Iterable[AblationRow]] = None,
) -> Path:
    """Write the ablation table to CSV and return the path."""
    df = ablation_to_dataframe(rows)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


__all__ = [
    "AblationRow",
    "DEFAULT_ABLATION",
    "ablation_to_dataframe",
    "write_ablation_csv",
]
