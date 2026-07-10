"""Runtime hardware detection: TPU -> GPU -> CPU fallback.

No hardcoded platform assumptions — the same notebook/script runs on a
Colab TPU runtime, a GPU runtime or a plain CPU machine.
"""

from __future__ import annotations

import tensorflow as tf


def detect_strategy(verbose: bool = True) -> tf.distribute.Strategy:
    """Return the best available distribution strategy.

    Order: TPU (``TPUStrategy``) -> one/multi GPU
    (``MirroredStrategy``) -> CPU (default strategy).
    """
    # --- TPU ---
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
        if verbose:
            print(f"[strategy] TPU detected: "
                  f"{strategy.num_replicas_in_sync} replicas")
        return strategy
    except (ValueError, tf.errors.NotFoundError, tf.errors.InvalidArgumentError):
        pass  # no TPU in this runtime

    # --- GPU ---
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        strategy = tf.distribute.MirroredStrategy()
        if verbose:
            print(f"[strategy] GPU detected: {len(gpus)} device(s), "
                  f"{strategy.num_replicas_in_sync} replicas")
        return strategy

    # --- CPU ---
    strategy = tf.distribute.get_strategy()
    if verbose:
        print("[strategy] no accelerator found — running on CPU")
    return strategy
