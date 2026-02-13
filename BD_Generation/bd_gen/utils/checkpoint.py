"""Checkpoint save/load utilities for training persistence."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: DictConfig,
    path: str | Path,
) -> None:
    """Save a training checkpoint to disk.

    Stores the model state dict, optimizer state dict, epoch number,
    and the full resolved Hydra config.  Parent directories are created
    automatically if they do not exist.

    Args:
        model: The model whose weights to save.
        optimizer: The optimizer whose state to save.
        epoch: Current epoch number (0-indexed).
        config: Full resolved Hydra :class:`DictConfig`.
        path: Destination file path (typically ``.pt``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "config": OmegaConf.to_container(config, resolve=True),
    }
    torch.save(checkpoint, path)
    logger.info("Checkpoint saved: %s (epoch %d)", path, epoch)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: str = "cpu",
) -> dict:
    """Load a training checkpoint from disk.

    Restores the model state dict into *model* and, optionally, the
    optimizer state dict into *optimizer*.  Returns a metadata dict
    containing the saved epoch and config so the caller can resume
    training from the correct point.

    Args:
        path: Path to the ``.pt`` checkpoint file.
        model: Model instance to load weights into.
        optimizer: If provided, also restore the optimizer state.
            Pass ``None`` for inference-only loading.
        device: Device to map tensors to (e.g. ``"cpu"``, ``"cuda"``).

    Returns:
        Dict with keys ``"epoch"`` (int) and ``"config"`` (plain dict).
    """
    path = Path(path)
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Model state loaded from %s", path)

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info("Optimizer state loaded from %s", path)

    return {
        "epoch": checkpoint["epoch"],
        "config": checkpoint["config"],
    }
