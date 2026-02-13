"""Reproducibility utilities: seed management for deterministic training."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for full reproducibility.

    Sets seeds for Python's ``random`` module, NumPy, and PyTorch (CPU
    and CUDA).  When CUDA is available, also enables deterministic cuDNN
    behaviour at the cost of a small performance penalty.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
