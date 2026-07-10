"""AFETSONAR TensorFlow/TPU port (tf-port branch).

Keras twin of the PyTorch training stack. Install requirements-tf.txt
into a SEPARATE environment (transformers 4.x — v5 dropped TF).

``TF_USE_LEGACY_KERAS`` must be set before transformers is imported;
we set it here defensively so ``import afetsonar_tf`` is always safe.
"""

import os

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

__version__ = "0.1.0"
