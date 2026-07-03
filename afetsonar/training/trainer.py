"""AfetsonarTrainer — incremental fine-tuning with catastrophic forgetting prevention.

Implements Experience Replay (Rolnick et al. 2019) to mitigate catastrophic
forgetting during continual learning: each effective mini-batch contains a
configurable fraction of old data alongside new disaster imagery.

Optionally applies SWA (Izmailov et al. 2018) in the final 20 % of training
epochs to find flatter minima and improve out-of-distribution generalisation.

Typical workflow::

    from afetsonar.training import AfetsonarTrainer

    trainer = AfetsonarTrainer("checkpoints/student_v1_best_ema.pth")

    # Step 1 — incorporate new data into a combined split CSV
    csvs = trainer.add_data(
        new_images_dir="new_dataset/images",
        new_labels_dir="new_dataset/labels",
        existing_csv="splits/train.csv",
    )

    # Step 2 — fine-tune from checkpoint with replay buffer
    result = trainer.resume_training(
        new_data_csv=csvs["train_csv"],
        val_csv=csvs["val_csv"],
        epochs=20,
        lr=1e-5,
    )
    print(result["best_val_miou"])

    # Step 3 — compare against the original checkpoint
    df = trainer.run_ablation(
        "splits/test.csv",
        experiment_name="v2_finetune",
        compare_checkpoint="checkpoints/student_v1_best_ema.pth",
    )

References
----------
- Rolnick et al. 2019 — Experience Replay for Continual Learning. NeurIPS.
- Izmailov et al. 2018 — Averaging Weights Leads to Wider Optima and Better
  Generalization. UAI 2018.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from afetsonar.config import DefaultConfig


class AfetsonarTrainer:
    """Continual learning trainer for incremental AFETSONAR fine-tuning.

    Args:
        checkpoint_path: Path to a teacher or student ``.pth`` checkpoint.
            The model is loaded *lazily* — no torch import happens at
            construction time.
        config: Hyper-parameter config.  Defaults to :class:`DefaultConfig`.
        device: Torch device string (``"cuda"`` / ``"cpu"`` / ``"auto"``).
        mode: ``"student"`` loads
            :class:`~afetsonar.models.StudentSiameseSegformer`;
            ``"teacher"`` loads
            :class:`~afetsonar.models.SiameseTeacherSegformerV3`.
        checkpoints_dir: Root directory for checkpoint files and the
            ``training_history.json`` log.

    Note:
        :meth:`add_data` requires only *pandas* and *pathlib* — it runs
        without torch.  :meth:`resume_training` and :meth:`run_ablation`
        require torch (and the full AFETSONAR dependency stack).
    """

    HISTORY_FILENAME = "training_history.json"

    def __init__(
        self,
        checkpoint_path: str,
        config: Optional[DefaultConfig] = None,
        device: str = "auto",
        mode: str = "student",
        checkpoints_dir: str = "checkpoints",
    ) -> None:
        if mode not in ("teacher", "student"):
            raise ValueError(
                f"mode must be 'teacher' or 'student', got {mode!r}"
            )
        self.checkpoint_path = checkpoint_path
        self.config = config or DefaultConfig()
        self.mode = mode
        self.checkpoints_dir = Path(checkpoints_dir)
        self._device_str = device

        # Lazy — populated on first call to _ensure_model_loaded()
        self._model: Any = None
        self._device_obj: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_data(
        self,
        new_images_dir: str,
        new_labels_dir: str,
        existing_csv: str,
        output_csv: Optional[str] = None,
        post_glob: str = "*_post_disaster.png",
        disaster_idx: int = 0,
        val_split: float = 0.1,
    ) -> Dict[str, str]:
        """Scan a directory for new images and combine with an existing CSV.

        Pairs files by stem: ``{base}_post_disaster.png`` ↔
        ``{base}_pre_disaster.png`` ↔ ``{base}_target.png``.
        Any image without a matching mask file is silently skipped.

        The combined CSV contains a ``data_source`` column (``"new"`` or
        ``"old"``).  :meth:`resume_training` uses this column to compute
        per-sample weights for the replay buffer.

        Args:
            new_images_dir: Directory containing new post (and optionally
                pre) disaster images.
            new_labels_dir: Directory containing ``*_target.png`` masks.
            existing_csv: Path to the current training split CSV.
            output_csv: Destination for the combined train CSV.  Defaults
                to ``{existing_csv_dir}/train_with_new_data.csv``.
            post_glob: Glob pattern for post-disaster images.
            disaster_idx: Disaster-type label assigned to all new samples.
            val_split: Fraction of *new* samples held out for validation.

        Returns:
            Dict with keys ``"train_csv"`` and ``"val_csv"``.

        Raises:
            ValueError: If no valid post-image / mask pairs are found.
        """
        import pandas as pd

        img_dir = Path(new_images_dir)
        lbl_dir = Path(new_labels_dir)

        post_files = sorted(img_dir.glob(post_glob))
        rows = []
        for post_path in post_files:
            base = post_path.stem.replace("_post_disaster", "")
            pre_path = img_dir / f"{base}_pre_disaster.png"
            mask_path = lbl_dir / f"{base}_target.png"

            if not mask_path.exists():
                continue

            rows.append({
                "post_path": str(post_path),
                "pre_path": str(pre_path) if pre_path.exists() else "",
                "mask_path": str(mask_path),
                "disaster_idx": disaster_idx,
                "filename": post_path.name,
                "data_source": "new",
            })

        if not rows:
            raise ValueError(
                f"No valid post/mask image pairs found.\n"
                f"  images: {img_dir}\n"
                f"  labels: {lbl_dir}\n"
                f"  pattern: {post_glob}"
            )

        # Shuffle and split new data into train / val
        new_df = (
            pd.DataFrame(rows)
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )
        split_at = max(1, int(len(new_df) * (1.0 - val_split)))
        new_train = new_df.iloc[:split_at].copy()
        new_val = new_df.iloc[split_at:].copy()

        # Load existing CSV and tag as old data
        existing_df = pd.read_csv(existing_csv)
        if "data_source" not in existing_df.columns:
            existing_df["data_source"] = "old"

        combined = pd.concat([existing_df, new_train], ignore_index=True)

        if output_csv is None:
            output_csv = str(
                Path(existing_csv).parent / "train_with_new_data.csv"
            )
        val_csv = str(
            Path(output_csv).parent / f"{Path(output_csv).stem}_val.csv"
        )

        combined.to_csv(output_csv, index=False)
        new_val.to_csv(val_csv, index=False)

        print(
            f"add_data: {len(rows)} new samples "
            f"({len(new_train)} train / {len(new_val)} val) "
            f"merged with {len(existing_df)} existing rows."
        )
        print(f"  train CSV -> {output_csv}")
        print(f"  val CSV   -> {val_csv}")
        return {"train_csv": output_csv, "val_csv": val_csv}

    def resume_training(
        self,
        new_data_csv: str,
        val_csv: str,
        epochs: int = 20,
        lr: float = 1e-5,
        replay_ratio: float = 0.2,
        old_data_csv: Optional[str] = None,
        save_every: int = 5,
        experiment_name: Optional[str] = None,
        use_swa: bool = False,
        batch_size: int = 4,
    ) -> Dict[str, Any]:
        """Fine-tune from the loaded checkpoint on new disaster imagery.

        Catastrophic forgetting is mitigated via Experience Replay: the
        combined CSV's ``data_source`` column is used to assign per-sample
        weights so that ~``replay_ratio`` of each effective batch comes
        from old data.

        Args:
            new_data_csv: Training CSV.  If it contains a ``data_source``
                column, rows marked ``"old"`` are down-weighted.
            val_csv: Validation CSV (no weighting applied).
            epochs: Fine-tuning epochs.
            lr: Peak learning rate.  Keep ≤ 1e-5 to avoid overwriting
                pre-trained features (warm-start regime).
            replay_ratio: Target fraction of each effective batch from old
                data.  0.0 disables replay.
            old_data_csv: Optional separate CSV for old replay data.  If
                ``None``, old rows are inferred from the ``data_source``
                column in ``new_data_csv``.
            save_every: Save a periodic checkpoint every N epochs.
            experiment_name: Human-readable run identifier.  Auto-generated
                from timestamp when ``None``.
            use_swa: Apply Stochastic Weight Averaging in the final 20 % of
                epochs and save an additional ``swa_model.pth`` checkpoint.
            batch_size: Batch size for both train and validation loaders.

        Returns:
            Dict with keys:

            - ``"experiment_name"`` — str.
            - ``"best_checkpoint"`` — path to the best EMA checkpoint.
            - ``"best_val_miou"`` — best val mIoU_no_bg achieved.
            - ``"epochs_trained"`` — int.
            - ``"history"`` — list of per-epoch metric dicts.
        """
        import pandas as pd
        import torch
        from afetsonar.models.ema import ModelEMA

        self._ensure_model_loaded()

        if experiment_name is None:
            experiment_name = f"resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # ---- Build combined training CSV ----
        train_df = pd.read_csv(new_data_csv)
        if old_data_csv is not None:
            old_df = pd.read_csv(old_data_csv)
            if "data_source" not in old_df.columns:
                old_df["data_source"] = "old"
            if "data_source" not in train_df.columns:
                train_df["data_source"] = "new"
            train_df = pd.concat([train_df, old_df], ignore_index=True)
        elif "data_source" not in train_df.columns:
            train_df["data_source"] = "new"

        has_old = (train_df["data_source"] == "old").any()
        sample_weights = (
            self._compute_replay_weights(train_df, replay_ratio)
            if replay_ratio > 0 and has_old
            else None
        )

        exp_dir = self.checkpoints_dir / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        combined_csv = str(exp_dir / "train_combined.csv")
        train_df.to_csv(combined_csv, index=False)

        # ---- Data loaders ----
        train_loader = self._build_loader(
            combined_csv, augment=True,
            sample_weights=sample_weights, batch_size=batch_size,
        )
        val_loader = self._build_loader(
            val_csv, augment=False, batch_size=batch_size,
        )

        # ---- Optimiser + cosine schedule with linear warmup ----
        optimizer = self._build_optimizer(lr)
        total_steps = len(train_loader) * epochs
        warmup_steps = len(train_loader) * min(2, epochs)
        scheduler = self._build_scheduler(optimizer, total_steps, warmup_steps)

        # ---- EMA ----
        ema = ModelEMA(self._model, decay=self.config.ema_decay)

        # ---- SWA (optional) ----
        swa_model = None
        swa_scheduler = None
        swa_start = int(0.8 * epochs) if use_swa else epochs + 1
        if use_swa:
            swa_model = torch.optim.swa_utils.AveragedModel(self._model)
            swa_scheduler = torch.optim.swa_utils.SWALR(
                optimizer, swa_lr=lr * 0.1
            )

        # ---- Mixed precision (CUDA only) ----
        use_amp = self.device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        criterion = self._build_criterion()
        best_miou = -1.0
        best_ckpt_path = ""
        epoch_history: List[Dict] = []

        # ---- Training loop ----
        self._model.train()
        for epoch in range(1, epochs + 1):
            train_metrics = self._train_one_epoch(
                train_loader, optimizer, criterion, ema, scaler, scheduler
            )

            # Validate with EMA shadow weights
            backup = ema.apply_to(self._model)
            self._model.eval()
            val_metrics = self._run_validation(val_loader)
            ema.restore(self._model, backup)
            self._model.train()

            # SWA update (last 20 % of epochs)
            if swa_model is not None and epoch >= swa_start:
                swa_model.update_parameters(self._model)
                swa_scheduler.step()

            row: Dict[str, Any] = {
                "epoch": epoch,
                **train_metrics,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            epoch_history.append(row)

            val_miou = float(val_metrics.get("miou_no_bg", 0.0))
            print(
                f"[{experiment_name}] {epoch:3d}/{epochs} | "
                f"loss={train_metrics['loss']:.4f} | "
                f"val_mIoU={val_miou:.4f} | "
                f"val_mF1={float(val_metrics.get('mf1', 0)):.4f}"
            )

            if epoch % save_every == 0:
                self._save_checkpoint(
                    epoch, row, experiment_name, f"epoch_{epoch:03d}"
                )

            if val_miou > best_miou:
                best_miou = val_miou
                best_ckpt_path = self._save_checkpoint(
                    epoch, row, experiment_name, "best_ema", ema=ema
                )

        # ---- SWA: update BN stats and save ----
        if swa_model is not None:
            torch.optim.swa_utils.update_bn(train_loader, swa_model)
            swa_path = str(exp_dir / "swa_model.pth")
            torch.save(
                {"model_state_dict": swa_model.module.state_dict()},
                swa_path,
            )
            print(f"SWA model saved -> {swa_path}")

        result: Dict[str, Any] = {
            "experiment_name": experiment_name,
            "best_checkpoint": best_ckpt_path,
            "best_val_miou": best_miou,
            "epochs_trained": epochs,
            "history": epoch_history,
        }

        self._append_history({
            "timestamp": datetime.now().isoformat(),
            "type": "resume_training",
            "experiment_name": experiment_name,
            "checkpoint_path": best_ckpt_path,
            "base_checkpoint": self.checkpoint_path,
            "mode": self.mode,
            "epochs": epochs,
            "lr": lr,
            "replay_ratio": replay_ratio,
            "new_data_csv": new_data_csv,
            "use_swa": use_swa,
            "best_val_miou": best_miou,
        })

        return result

    def run_ablation(
        self,
        test_csv: str,
        experiment_name: str,
        compare_checkpoint: Optional[str] = None,
    ) -> Any:  # pd.DataFrame
        """Evaluate the loaded checkpoint and optionally compare with another.

        Runs full segmentation evaluation on ``test_csv``, records results
        in ``training_history.json``, and returns a tidy comparison table.

        Args:
            test_csv: Test split CSV (same schema as training CSVs).
            experiment_name: Label for this evaluation run.
            compare_checkpoint: Optional second checkpoint to evaluate on
                the same test set for side-by-side comparison.

        Returns:
            :class:`pandas.DataFrame` with columns ``experiment``,
            ``checkpoint``, ``miou_no_bg``, ``mf1``, ``accuracy``.
        """
        import pandas as pd

        self._ensure_model_loaded()
        test_loader = self._build_loader(test_csv, augment=False, batch_size=4)

        rows: List[Dict] = []

        self._model.eval()
        scores = self._run_validation(test_loader)
        rows.append({
            "experiment": experiment_name,
            "checkpoint": Path(self.checkpoint_path).name,
            **{k: round(v, 4) for k, v in scores.items()
               if isinstance(v, float)},
        })

        if compare_checkpoint is not None:
            saved = self._model
            try:
                self._model = self._load_model(compare_checkpoint)
                self._model.eval()
                cmp_scores = self._run_validation(test_loader)
                rows.append({
                    "experiment": f"{experiment_name}_base",
                    "checkpoint": Path(compare_checkpoint).name,
                    **{k: round(v, 4) for k, v in cmp_scores.items()
                       if isinstance(v, float)},
                })
            finally:
                self._model = saved

        df = pd.DataFrame(rows)

        self._append_history({
            "timestamp": datetime.now().isoformat(),
            "type": "ablation",
            "experiment_name": experiment_name,
            "checkpoint_path": self.checkpoint_path,
            "test_csv": test_csv,
            "metrics": {k: v for k, v in rows[0].items()
                        if isinstance(v, float)},
        })

        print_cols = [c for c in ("experiment", "checkpoint", "miou_no_bg", "mf1", "accuracy") if c in df.columns]
        print(df[print_cols].to_string(index=False))
        return df

    # ------------------------------------------------------------------
    # Device property (lazy torch import)
    # ------------------------------------------------------------------

    @property
    def device(self) -> Any:
        if self._device_obj is None:
            import torch
            self._device_obj = (
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if self._device_str == "auto"
                else torch.device(self._device_str)
            )
        return self._device_obj

    # ------------------------------------------------------------------
    # Private — model management
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self) -> None:
        if self._model is None:
            self._model = self._load_model(self.checkpoint_path)

    def _load_model(self, checkpoint_path: str) -> Any:
        import torch

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            # Prefer baked-in EMA weights (saved by _save_checkpoint)
            if "ema_state" in checkpoint:
                state_dict = checkpoint["ema_state"]
            else:
                for key in ("model_state_dict", "state_dict", "model"):
                    if key in checkpoint:
                        state_dict = checkpoint[key]
                        break

        if self.mode == "teacher":
            from afetsonar.models.teacher import SiameseTeacherSegformerV3
            model = SiameseTeacherSegformerV3(
                num_damage_classes=self.config.num_classes,
                num_disaster_classes=self.config.num_disaster_classes,
                pretrained=False,
            )
        else:
            from afetsonar.models.student import StudentSiameseSegformer
            model = StudentSiameseSegformer(
                num_damage_classes=self.config.num_classes,
                num_disaster_classes=self.config.num_disaster_classes,
                pretrained=False,
            )

        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        return model

    def _build_criterion(self) -> Any:
        from afetsonar.losses.combo import ComboDamageLossV3
        return ComboDamageLossV3(
            num_classes=self.config.num_classes,
            class_weights=self.config.class_weights,
        )

    def _build_optimizer(self, lr: float) -> Any:
        import torch
        return torch.optim.AdamW(
            self._model.parameters(), lr=lr, weight_decay=1e-4
        )

    def _build_scheduler(
        self,
        optimizer: Any,
        total_steps: int,
        warmup_steps: int,
    ) -> Any:
        """Linear warmup → cosine annealing, stepped once per batch."""
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        import torch
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def _build_loader(
        self,
        csv_path: str,
        augment: bool = False,
        sample_weights: Optional[np.ndarray] = None,
        batch_size: int = 4,
        num_workers: int = 0,
    ) -> Any:
        import torch
        from torch.utils.data import DataLoader, WeightedRandomSampler
        from afetsonar.data.dataset import XBDDatasetV2
        from afetsonar.data.augmentations import (
            get_train_augmentation_v2,
            get_val_augmentation_v2,
        )

        aug = (
            get_train_augmentation_v2(
                image_size=self.config.image_size, mode="teacher"
            )
            if augment
            else get_val_augmentation_v2(
                image_size=self.config.image_size, mode="teacher"
            )
        )

        # Both SiameseTeacherSegformerV3 and StudentSiameseSegformer accept
        # 6-channel (pre + post) input, so always use "teacher" dataset mode.
        ds = XBDDatasetV2(
            csv_path=csv_path,
            mode="teacher",
            augmentation=aug,
            image_size=self.config.image_size,
        )

        if sample_weights is not None:
            sampler = WeightedRandomSampler(
                weights=torch.from_numpy(sample_weights).float(),
                num_samples=len(ds),
                replacement=True,
            )
            return DataLoader(
                ds,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
            )
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=augment,
            num_workers=num_workers,
        )

    # ------------------------------------------------------------------
    # Private — training loop
    # ------------------------------------------------------------------

    def _compute_replay_weights(
        self,
        df: Any,  # pd.DataFrame
        replay_ratio: float,
    ) -> np.ndarray:
        """Per-sample weights so ~``replay_ratio`` of each batch is old data.

        Solves:  n_old * w_old / (n_new * 1 + n_old * w_old) = replay_ratio
        giving:  w_old = replay_ratio * n_new / ((1 - replay_ratio) * n_old)
        """
        is_new = (df["data_source"] == "new").values
        n_new = int(is_new.sum())
        n_old = int((~is_new).sum())

        weights = np.ones(len(df), dtype=np.float32)
        if n_old == 0 or n_new == 0 or replay_ratio <= 0:
            return weights

        old_weight = (replay_ratio * n_new) / ((1.0 - replay_ratio) * n_old)
        weights[~is_new] = float(old_weight)
        return weights

    def _train_one_epoch(
        self,
        loader: Any,
        optimizer: Any,
        criterion: Any,
        ema: Any,
        scaler: Any,
        scheduler: Any,
    ) -> Dict[str, float]:
        import torch

        self._model.train()
        total_loss = 0.0

        for batch in loader:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            optimizer.zero_grad()

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self._model(images)
                    logits = outputs["damage_logits"]
                    if isinstance(logits, (list, tuple)):
                        logits = logits[0]
                    loss_dict = criterion(logits, masks)
                scaler.scale(loss_dict["total"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = self._model(images)
                logits = outputs["damage_logits"]
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                loss_dict = criterion(logits, masks)
                loss_dict["total"].backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            ema.update(self._model)
            total_loss += loss_dict["total"].item()

        return {"loss": total_loss / max(len(loader), 1)}

    def _run_validation(self, loader: Any) -> Dict[str, float]:
        import torch
        from afetsonar.evaluation.metrics import SegmentationMetrics

        criterion = self._build_criterion()
        seg = SegmentationMetrics(num_classes=self.config.num_classes)
        total_loss = 0.0

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)
                outputs = self._model(images)
                logits = outputs["damage_logits"]
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                loss_dict = criterion(logits, masks)
                total_loss += loss_dict["total"].item()
                seg.update(logits.argmax(dim=1), masks)

        scores = seg.compute()
        scores["val_loss"] = total_loss / max(len(loader), 1)
        return scores

    # ------------------------------------------------------------------
    # Private — checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, Any],
        experiment_name: str,
        suffix: str,
        ema: Optional[Any] = None,
    ) -> str:
        import torch

        exp_dir = self.checkpoints_dir / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        path = str(exp_dir / f"{experiment_name}_{suffix}.pth")

        state: Dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": self._model.state_dict(),
            "metrics": {
                k: v for k, v in metrics.items()
                if isinstance(v, (int, float, str))
            },
            "mode": self.mode,
            "base_checkpoint": self.checkpoint_path,
        }
        if ema is not None:
            # Store EMA shadow weights as the primary model weights so the
            # checkpoint can be loaded directly by AfetsonarPipeline.
            state["ema_state"] = {k: v.cpu() for k, v in ema.shadow.items()}

        torch.save(state, path)
        return path

    def _load_history(self) -> List[Dict]:
        path = self.checkpoints_dir / self.HISTORY_FILENAME
        if not path.exists():
            return []
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []

    def _append_history(self, entry: Dict) -> None:
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        history = self._load_history()
        history.append(entry)
        path = self.checkpoints_dir / self.HISTORY_FILENAME
        with open(path, "w") as f:
            json.dump(history, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"AfetsonarTrainer("
            f"mode={self.mode!r}, "
            f"checkpoint={Path(self.checkpoint_path).name!r}, "
            f"model_loaded={self._model is not None})"
        )
