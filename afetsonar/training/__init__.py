"""AFETSONAR incremental training utilities.

Public API::

    from afetsonar.training import AfetsonarTrainer

    trainer = AfetsonarTrainer("checkpoints/student_v1_best_ema.pth")
    csvs = trainer.add_data("new_images/", "new_labels/", "splits/train.csv")
    result = trainer.resume_training(csvs["train_csv"], csvs["val_csv"], epochs=20)
    df = trainer.run_ablation("splits/test.csv", "v2_finetune")
"""

from afetsonar.training.trainer import AfetsonarTrainer

__all__ = ["AfetsonarTrainer"]
