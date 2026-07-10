"""Keras models for the AFETSONAR TF port."""

from afetsonar_tf.models.ema_tf import EmaShadows

__all__ = ["EmaShadows", "build_tf_teacher"]


def build_tf_teacher(*args, **kwargs):
    """Lazy import so ``afetsonar_tf.losses`` stays usable without HF."""
    from afetsonar_tf.models.teacher_tf import build_tf_teacher as _b
    return _b(*args, **kwargs)
