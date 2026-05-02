"""Tests for AfetsonarTrainer.

Groups:
- TestConstruction       : no torch needed
- TestAddData            : no torch needed (file scanning + CSV ops)
- TestReplayWeights      : no torch needed (numpy arithmetic)
- TestHistoryManagement  : no torch needed (JSON read/write)
- TestResumeTraining     : torch required (smoke tests, auto-skipped if absent)
- TestRunAblation        : torch required (smoke tests, auto-skipped if absent)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from afetsonar.training.trainer import AfetsonarTrainer

# ------------------------------------------------------------------
# Torch availability — checked once at import time, not via
# importorskip (which would skip the whole module when torch is absent)
# ------------------------------------------------------------------
try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    _torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

_requires_torch = pytest.mark.skipif(
    not _TORCH_AVAILABLE,
    reason="torch not installed — skipping trainer training tests",
)


# ============================================================
# Helpers
# ============================================================

def _make_trainer(tmp_path: Path, mode: str = "student") -> AfetsonarTrainer:
    return AfetsonarTrainer(
        checkpoint_path=str(tmp_path / "dummy.pth"),
        mode=mode,
        checkpoints_dir=str(tmp_path / "ckpts"),
    )


def _make_existing_csv(tmp_path: Path) -> str:
    df = pd.DataFrame([{
        "post_path": str(tmp_path / "old_post.png"),
        "pre_path":  str(tmp_path / "old_pre.png"),
        "mask_path": str(tmp_path / "old_mask.png"),
        "disaster_idx": 0,
        "filename": "old_post.png",
    }])
    p = str(tmp_path / "train.csv")
    df.to_csv(p, index=False)
    return p


def _make_fake_images(img_dir: Path, lbl_dir: Path, n: int = 4) -> None:
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (img_dir / f"event_{i:06d}_post_disaster.png").touch()
        (img_dir / f"event_{i:06d}_pre_disaster.png").touch()
        (lbl_dir / f"event_{i:06d}_target.png").touch()


def _make_tiny_csv(tmp_path: Path, n: int = 4, source: str = "new") -> str:
    df = pd.DataFrame([{
        "post_path": str(tmp_path / f"p{i}.png"),
        "pre_path":  str(tmp_path / f"r{i}.png"),
        "mask_path": str(tmp_path / f"m{i}.png"),
        "disaster_idx": 0,
        "filename": f"p{i}.png",
        "data_source": source,
    } for i in range(n)])
    path = str(tmp_path / f"split_{source}.csv")
    df.to_csv(path, index=False)
    return path


def _fake_batch():
    """One DataLoader batch with 2 samples, 64×64, 6 channels."""
    import torch
    return {
        "image": torch.zeros(2, 6, 64, 64),
        "mask":  torch.zeros(2, 64, 64, dtype=torch.long),
        "disaster_idx": torch.zeros(2, dtype=torch.long),
        "filename": ["a.png", "b.png"],
    }


def _make_wrapped_conv():
    """Minimal model that mimics AfetsonarPipeline output dict format."""
    import torch

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(6, 6, 1)

        def forward(self, x):
            logits = self.conv(x)
            return {
                "damage_logits": logits,
                "change_logits": torch.zeros(
                    x.shape[0], 2, x.shape[2], x.shape[3]
                ),
                "disaster_logits": torch.zeros(x.shape[0], 5),
            }

    return _Model()


# ============================================================
# Construction
# ============================================================

class TestConstruction:
    def test_default_mode_is_student(self, tmp_path):
        t = AfetsonarTrainer(str(tmp_path / "ckpt.pth"))
        assert t.mode == "student"

    def test_teacher_mode_accepted(self, tmp_path):
        t = AfetsonarTrainer(str(tmp_path / "ckpt.pth"), mode="teacher")
        assert t.mode == "teacher"

    def test_invalid_mode_raises(self, tmp_path):
        with pytest.raises(ValueError, match="mode must be"):
            AfetsonarTrainer(str(tmp_path / "ckpt.pth"), mode="invalid")

    def test_model_not_loaded_at_construction(self, tmp_path):
        assert _make_trainer(tmp_path)._model is None

    def test_repr_shows_mode_and_checkpoint(self, tmp_path):
        t = AfetsonarTrainer(
            str(tmp_path / "my_model.pth"), mode="student",
            checkpoints_dir=str(tmp_path / "ckpts"),
        )
        r = repr(t)
        assert "student" in r
        assert "my_model.pth" in r
        assert "model_loaded=False" in r

    def test_repr_updates_when_model_injected(self, tmp_path):
        t = _make_trainer(tmp_path)
        t._model = object()
        assert "model_loaded=True" in repr(t)


# ============================================================
# add_data
# ============================================================

class TestAddData:
    def test_returns_dict_with_expected_keys(self, tmp_path):
        img, lbl = tmp_path / "img", tmp_path / "lbl"
        _make_fake_images(img, lbl, n=5)
        result = _make_trainer(tmp_path).add_data(
            str(img), str(lbl), _make_existing_csv(tmp_path)
        )
        assert set(result) == {"train_csv", "val_csv"}

    def test_output_csvs_are_created(self, tmp_path):
        img, lbl = tmp_path / "img", tmp_path / "lbl"
        _make_fake_images(img, lbl, n=5)
        result = _make_trainer(tmp_path).add_data(
            str(img), str(lbl), _make_existing_csv(tmp_path)
        )
        assert Path(result["train_csv"]).exists()
        assert Path(result["val_csv"]).exists()

    def test_combined_csv_has_data_source_column(self, tmp_path):
        img, lbl = tmp_path / "img", tmp_path / "lbl"
        _make_fake_images(img, lbl, n=5)
        result = _make_trainer(tmp_path).add_data(
            str(img), str(lbl), _make_existing_csv(tmp_path)
        )
        combined = pd.read_csv(result["train_csv"])
        assert "data_source" in combined.columns
        assert set(combined["data_source"].unique()).issubset({"new", "old"})

    def test_old_rows_preserved_in_combined(self, tmp_path):
        img, lbl = tmp_path / "img", tmp_path / "lbl"
        _make_fake_images(img, lbl, n=5)
        result = _make_trainer(tmp_path).add_data(
            str(img), str(lbl), _make_existing_csv(tmp_path)
        )
        combined = pd.read_csv(result["train_csv"])
        assert (combined["data_source"] == "old").sum() >= 1

    def test_new_rows_present_when_val_split_zero(self, tmp_path):
        img, lbl = tmp_path / "img", tmp_path / "lbl"
        _make_fake_images(img, lbl, n=6)
        result = _make_trainer(tmp_path).add_data(
            str(img), str(lbl), _make_existing_csv(tmp_path), val_split=0.0
        )
        combined = pd.read_csv(result["train_csv"])
        assert (combined["data_source"] == "new").sum() == 6

    def test_no_matching_files_raises(self, tmp_path):
        img, lbl = tmp_path / "img", tmp_path / "lbl"
        img.mkdir(); lbl.mkdir()
        with pytest.raises(ValueError, match="No valid post/mask"):
            _make_trainer(tmp_path).add_data(
                str(img), str(lbl), _make_existing_csv(tmp_path)
            )

    def test_custom_output_csv_path(self, tmp_path):
        img, lbl = tmp_path / "img", tmp_path / "lbl"
        _make_fake_images(img, lbl, n=3)
        custom = str(tmp_path / "my_combined.csv")
        result = _make_trainer(tmp_path).add_data(
            str(img), str(lbl), _make_existing_csv(tmp_path),
            output_csv=custom,
        )
        assert result["train_csv"] == custom
        assert Path(custom).exists()

    def test_val_split_fraction(self, tmp_path):
        """10 new images, val_split=0.2 → 2 val rows."""
        img, lbl = tmp_path / "img", tmp_path / "lbl"
        _make_fake_images(img, lbl, n=10)
        result = _make_trainer(tmp_path).add_data(
            str(img), str(lbl), _make_existing_csv(tmp_path), val_split=0.2
        )
        assert len(pd.read_csv(result["val_csv"])) == 2

    def test_images_without_mask_are_skipped(self, tmp_path):
        img, lbl = tmp_path / "img", tmp_path / "lbl"
        img.mkdir(); lbl.mkdir()
        for i in range(3):
            (img / f"ev_{i:04d}_post_disaster.png").touch()
            (img / f"ev_{i:04d}_pre_disaster.png").touch()
        for i in range(2):  # only 2 masks
            (lbl / f"ev_{i:04d}_target.png").touch()
        result = _make_trainer(tmp_path).add_data(
            str(img), str(lbl), _make_existing_csv(tmp_path), val_split=0.0
        )
        combined = pd.read_csv(result["train_csv"])
        assert (combined["data_source"] == "new").sum() == 2

    def test_existing_csv_without_source_col_tagged_old(self, tmp_path):
        img, lbl = tmp_path / "img", tmp_path / "lbl"
        _make_fake_images(img, lbl, n=3)
        df = pd.DataFrame([{
            "post_path": "a.png", "pre_path": "b.png",
            "mask_path": "c.png", "disaster_idx": 0, "filename": "a.png",
        }])
        csv = str(tmp_path / "no_col.csv")
        df.to_csv(csv, index=False)
        result = _make_trainer(tmp_path).add_data(
            str(img), str(lbl), csv, val_split=0.0
        )
        combined = pd.read_csv(result["train_csv"])
        assert (combined["data_source"] == "old").sum() == 1

    def test_disaster_idx_applied_to_new_rows(self, tmp_path):
        img, lbl = tmp_path / "img", tmp_path / "lbl"
        _make_fake_images(img, lbl, n=3)
        result = _make_trainer(tmp_path).add_data(
            str(img), str(lbl), _make_existing_csv(tmp_path),
            disaster_idx=3, val_split=0.0,
        )
        combined = pd.read_csv(result["train_csv"])
        new_rows = combined[combined["data_source"] == "new"]
        assert (new_rows["disaster_idx"] == 3).all()


# ============================================================
# Replay weight computation
# ============================================================

class TestReplayWeights:
    def test_all_new_returns_uniform(self, tmp_path):
        df = pd.DataFrame({"data_source": ["new"] * 10})
        w = _make_trainer(tmp_path)._compute_replay_weights(df, 0.2)
        np.testing.assert_array_equal(w, np.ones(10, dtype=np.float32))

    def test_effective_old_fraction_correct(self, tmp_path):
        n_new, n_old = 8, 2
        df = pd.DataFrame({"data_source": ["new"] * n_new + ["old"] * n_old})
        w = _make_trainer(tmp_path)._compute_replay_weights(df, 0.2)
        old_w = float(w[n_new])
        eff_old = (n_old * old_w) / (n_new * 1.0 + n_old * old_w)
        assert abs(eff_old - 0.2) < 0.01

    def test_zero_ratio_returns_uniform(self, tmp_path):
        df = pd.DataFrame({"data_source": ["new"] * 5 + ["old"] * 5})
        w = _make_trainer(tmp_path)._compute_replay_weights(df, 0.0)
        np.testing.assert_array_equal(w, np.ones(10, dtype=np.float32))

    def test_no_old_rows_returns_uniform(self, tmp_path):
        df = pd.DataFrame({"data_source": ["new"] * 8})
        w = _make_trainer(tmp_path)._compute_replay_weights(df, 0.3)
        np.testing.assert_array_equal(w, np.ones(8, dtype=np.float32))

    def test_high_ratio_up_weights_old(self, tmp_path):
        n_new, n_old = 90, 10
        df = pd.DataFrame({"data_source": ["new"] * n_new + ["old"] * n_old})
        w = _make_trainer(tmp_path)._compute_replay_weights(df, 0.5)
        assert float(w[n_new]) > 1.0

    def test_output_dtype_float32(self, tmp_path):
        df = pd.DataFrame({"data_source": ["new"] * 4 + ["old"] * 4})
        w = _make_trainer(tmp_path)._compute_replay_weights(df, 0.2)
        assert w.dtype == np.float32


# ============================================================
# History management
# ============================================================

class TestHistoryManagement:
    def test_empty_history_is_empty_list(self, tmp_path):
        assert _make_trainer(tmp_path)._load_history() == []

    def test_append_creates_file(self, tmp_path):
        t = _make_trainer(tmp_path)
        t._append_history({"experiment": "e1"})
        path = Path(str(tmp_path / "ckpts")) / "training_history.json"
        assert path.exists()

    def test_append_and_load_roundtrip(self, tmp_path):
        t = _make_trainer(tmp_path)
        t._append_history({"experiment": "run1", "val_miou": 0.42})
        t._append_history({"experiment": "run2", "val_miou": 0.45})
        history = t._load_history()
        assert len(history) == 2
        assert history[0]["experiment"] == "run1"
        assert history[1]["val_miou"] == 0.45

    def test_history_is_valid_json(self, tmp_path):
        t = _make_trainer(tmp_path)
        t._append_history({"x": 1})
        path = Path(str(tmp_path / "ckpts")) / "training_history.json"
        data = json.loads(path.read_text())
        assert isinstance(data, list)

    def test_corrupted_json_returns_empty(self, tmp_path):
        t = _make_trainer(tmp_path)
        ckpts = tmp_path / "ckpts"
        ckpts.mkdir(parents=True, exist_ok=True)
        (ckpts / "training_history.json").write_text("not json {{{")
        assert t._load_history() == []

    def test_multiple_appends_cumulative(self, tmp_path):
        t = _make_trainer(tmp_path)
        for i in range(5):
            t._append_history({"step": i})
        assert len(t._load_history()) == 5


# ============================================================
# Torch-dependent smoke tests
# ============================================================

@pytest.fixture
def mock_trainer(tmp_path):
    """AfetsonarTrainer pre-loaded with a tiny real CPU model."""
    import torch
    t = AfetsonarTrainer(
        checkpoint_path=str(tmp_path / "dummy.pth"),
        mode="student",
        checkpoints_dir=str(tmp_path / "ckpts"),
    )
    t._model = _make_wrapped_conv()
    t._device_obj = torch.device("cpu")
    return t


@_requires_torch
class TestResumeTraining:
    def test_returns_expected_keys(self, tmp_path, mock_trainer):
        train_csv = _make_tiny_csv(tmp_path, n=4)
        val_csv   = _make_tiny_csv(tmp_path, n=2)
        with patch.object(mock_trainer, "_build_loader", return_value=[_fake_batch()]):
            result = mock_trainer.resume_training(
                train_csv, val_csv, epochs=1, replay_ratio=0.0,
                experiment_name="smoke",
            )
        assert set(result) >= {
            "experiment_name", "best_checkpoint",
            "best_val_miou", "epochs_trained", "history",
        }

    def test_history_length_equals_epochs(self, tmp_path, mock_trainer):
        train_csv = _make_tiny_csv(tmp_path, n=4)
        val_csv   = _make_tiny_csv(tmp_path, n=2)
        with patch.object(mock_trainer, "_build_loader", return_value=[_fake_batch()]):
            result = mock_trainer.resume_training(
                train_csv, val_csv, epochs=3, replay_ratio=0.0,
                experiment_name="hist",
            )
        assert len(result["history"]) == 3

    def test_best_checkpoint_saved_to_disk(self, tmp_path, mock_trainer):
        train_csv = _make_tiny_csv(tmp_path, n=4)
        val_csv   = _make_tiny_csv(tmp_path, n=2)
        with patch.object(mock_trainer, "_build_loader", return_value=[_fake_batch()]):
            result = mock_trainer.resume_training(
                train_csv, val_csv, epochs=1, replay_ratio=0.0,
                experiment_name="save_test",
            )
        assert Path(result["best_checkpoint"]).exists()

    def test_history_json_updated(self, tmp_path, mock_trainer):
        train_csv = _make_tiny_csv(tmp_path, n=4)
        val_csv   = _make_tiny_csv(tmp_path, n=2)
        with patch.object(mock_trainer, "_build_loader", return_value=[_fake_batch()]):
            mock_trainer.resume_training(
                train_csv, val_csv, epochs=1, replay_ratio=0.0,
                experiment_name="json_test",
            )
        history = mock_trainer._load_history()
        assert len(history) == 1
        assert history[0]["experiment_name"] == "json_test"
        assert history[0]["type"] == "resume_training"

    def test_epoch_rows_contain_loss_and_val_keys(self, tmp_path, mock_trainer):
        train_csv = _make_tiny_csv(tmp_path, n=4)
        val_csv   = _make_tiny_csv(tmp_path, n=2)
        with patch.object(mock_trainer, "_build_loader", return_value=[_fake_batch()]):
            result = mock_trainer.resume_training(
                train_csv, val_csv, epochs=2, replay_ratio=0.0,
                experiment_name="metrics_test",
            )
        for row in result["history"]:
            assert "epoch" in row
            assert "loss" in row

    def test_periodic_checkpoint_saved(self, tmp_path, mock_trainer):
        train_csv = _make_tiny_csv(tmp_path, n=4)
        val_csv   = _make_tiny_csv(tmp_path, n=2)
        with patch.object(mock_trainer, "_build_loader", return_value=[_fake_batch()]):
            mock_trainer.resume_training(
                train_csv, val_csv, epochs=4, save_every=2,
                replay_ratio=0.0, experiment_name="periodic",
            )
        ckpt_dir = tmp_path / "ckpts" / "periodic"
        assert len(list(ckpt_dir.glob("*epoch_002*"))) == 1

    def test_combined_csv_saved_in_experiment_dir(self, tmp_path, mock_trainer):
        train_csv = _make_tiny_csv(tmp_path, n=4)
        val_csv   = _make_tiny_csv(tmp_path, n=2)
        with patch.object(mock_trainer, "_build_loader", return_value=[_fake_batch()]):
            mock_trainer.resume_training(
                train_csv, val_csv, epochs=1, replay_ratio=0.0,
                experiment_name="csvcheck",
            )
        combined = tmp_path / "ckpts" / "csvcheck" / "train_combined.csv"
        assert combined.exists()


@_requires_torch
class TestRunAblation:
    def test_returns_dataframe(self, tmp_path, mock_trainer):
        test_csv = _make_tiny_csv(tmp_path, n=2)
        with patch.object(mock_trainer, "_build_loader", return_value=[_fake_batch()]):
            df = mock_trainer.run_ablation(test_csv, "ablation_smoke")
        assert hasattr(df, "columns")  # it's a DataFrame

    def test_dataframe_has_expected_columns(self, tmp_path, mock_trainer):
        test_csv = _make_tiny_csv(tmp_path, n=2)
        with patch.object(mock_trainer, "_build_loader", return_value=[_fake_batch()]):
            df = mock_trainer.run_ablation(test_csv, "col_test")
        for col in ("experiment", "checkpoint"):
            assert col in df.columns

    def test_single_checkpoint_one_row(self, tmp_path, mock_trainer):
        test_csv = _make_tiny_csv(tmp_path, n=2)
        with patch.object(mock_trainer, "_build_loader", return_value=[_fake_batch()]):
            df = mock_trainer.run_ablation(test_csv, "single_row")
        assert len(df) == 1

    def test_ablation_logged_to_history(self, tmp_path, mock_trainer):
        test_csv = _make_tiny_csv(tmp_path, n=2)
        with patch.object(mock_trainer, "_build_loader", return_value=[_fake_batch()]):
            mock_trainer.run_ablation(test_csv, "log_test")
        history = mock_trainer._load_history()
        assert len(history) == 1
        assert history[0]["type"] == "ablation"
        assert history[0]["experiment_name"] == "log_test"
