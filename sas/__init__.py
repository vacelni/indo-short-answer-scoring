"""Automatic scoring of Indonesian short answers against a teacher's reference answer."""

import os

# transformers would also load TensorFlow when it happens to be installed; this project is PyTorch-only.
os.environ.setdefault("USE_TF", "0")
