"""Logging utilities: wandb initialisation, metric logging, git hash."""

from __future__ import annotations

import logging
import subprocess

import wandb
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def get_git_commit() -> str:
    """Return the current HEAD commit hash (40-char hex string).

    Falls back to ``"unknown"`` when *git* is not installed or the
    working directory is not inside a repository.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def init_wandb(config: DictConfig) -> None:
    """Initialise a wandb run with the full Hydra config and git hash.

    Reads ``config.wandb.project``, ``config.wandb.entity``, and
    ``config.wandb.mode`` to configure the run.  The complete resolved
    config is uploaded so every run is fully reproducible from its
    wandb page.

    Args:
        config: Full resolved Hydra :class:`DictConfig`.
    """
    git_hash = get_git_commit()

    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.get("entity", None),
        name=config.experiment_name,
        mode=config.wandb.mode,
        config=OmegaConf.to_container(config, resolve=True),
    )
    wandb.config.update({"git_commit": git_hash}, allow_val_change=True)
    logger.info(
        "wandb initialised: project=%s, mode=%s, git=%s",
        config.wandb.project,
        config.wandb.mode,
        git_hash[:8],
    )


def log_metrics(metrics: dict, step: int) -> None:
    """Log a dictionary of metrics to the active wandb run.

    Args:
        metrics: Key-value pairs (e.g. ``{"loss": 0.5, "lr": 3e-4}``).
        step: Global step number for x-axis alignment.
    """
    wandb.log(metrics, step=step)
