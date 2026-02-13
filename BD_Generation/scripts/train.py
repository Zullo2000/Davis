"""MDLM training loop for bubble diagram generation.

Uses Hydra's Compose API (not the ``@hydra.main`` decorator) to avoid
an argparse incompatibility with Python 3.14.  CLI overrides are still
supported via ``sys.argv``.

Usage:
    python scripts/train.py                             # defaults
    python scripts/train.py model=base                  # larger model
    python scripts/train.py wandb.mode=disabled         # no wandb
    python scripts/train.py training.epochs=1           # quick debug
    python scripts/train.py data.batch_size=32          # smaller batch for CPU
"""

from __future__ import annotations

import logging
import platform
import sys
from datetime import datetime
from pathlib import Path

import torch
import wandb
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure BD_Generation is on sys.path when running as a script
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from bd_gen.data.dataset import BubbleDiagramDataset  # noqa: E402
from bd_gen.data.vocab import VocabConfig  # noqa: E402
from bd_gen.diffusion.forward_process import forward_mask  # noqa: E402
from bd_gen.diffusion.loss import ELBOLoss  # noqa: E402
from bd_gen.diffusion.noise_schedule import get_noise  # noqa: E402
from bd_gen.diffusion.sampling import sample  # noqa: E402
from bd_gen.model.denoiser import BDDenoiser  # noqa: E402
from bd_gen.utils.checkpoint import save_checkpoint  # noqa: E402
from bd_gen.utils.logging_utils import init_wandb, log_metrics  # noqa: E402
from bd_gen.utils.seed import set_seed  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading (Hydra Compose API)
# ---------------------------------------------------------------------------


def _load_config(overrides: list[str] | None = None) -> DictConfig:
    """Load and compose the Hydra config with CLI overrides.

    Uses the Compose API instead of ``@hydra.main`` to avoid an
    argparse incompatibility with Python >= 3.14.
    """
    config_dir = str((_PROJECT_ROOT / "configs").resolve())

    # Clear any previous Hydra state (e.g. from tests or re-runs)
    GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config", overrides=overrides or [])

    return cfg


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------


def _build_lr_lambda(warmup_steps: int):
    """Return a lambda for :class:`LambdaLR` implementing linear warmup.

    LR grows linearly from 0 to the base LR over *warmup_steps*
    optimiser steps, then stays constant.
    """

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return 1.0

    return lr_lambda


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate(
    model: BDDenoiser,
    val_loader: DataLoader,
    criterion: ELBOLoss,
    noise_schedule: torch.nn.Module,
    vocab_config: VocabConfig,
    device: str,
) -> dict[str, float]:
    """Run one full validation pass and return metrics.

    Returns:
        Dict with ``val/loss``, ``val/node_accuracy``,
        ``val/edge_accuracy``.
    """
    model.eval()
    val_loss_sum = 0.0
    val_batches = 0

    node_correct = 0
    node_total = 0
    edge_correct = 0
    edge_total = 0

    n_max = vocab_config.n_max

    with torch.no_grad():
        for batch in val_loader:
            tokens = batch["tokens"].to(device)
            pad_mask = batch["pad_mask"].to(device)

            B = tokens.size(0)
            t = torch.rand(B, device=device)
            t = torch.clamp(t, min=1e-5, max=1.0)

            x_t, mask_indicators = forward_mask(
                tokens, pad_mask, t, noise_schedule, vocab_config,
            )
            node_logits, edge_logits = model(x_t, pad_mask, t)
            loss = criterion(
                node_logits, edge_logits,
                tokens, x_t, pad_mask, mask_indicators,
                t, noise_schedule,
            )

            val_loss_sum += loss.item()
            val_batches += 1

            # --- Per-class accuracy at masked positions ---
            node_mask = mask_indicators[:, :n_max] & pad_mask[:, :n_max]
            if node_mask.any():
                node_pred = node_logits.argmax(dim=-1)
                node_tgt = tokens[:, :n_max]
                node_correct += ((node_pred == node_tgt) & node_mask).sum().item()
                node_total += node_mask.sum().item()

            edge_mask = mask_indicators[:, n_max:] & pad_mask[:, n_max:]
            if edge_mask.any():
                edge_pred = edge_logits.argmax(dim=-1)
                edge_tgt = tokens[:, n_max:]
                edge_correct += ((edge_pred == edge_tgt) & edge_mask).sum().item()
                edge_total += edge_mask.sum().item()

    avg_val_loss = val_loss_sum / max(val_batches, 1)
    node_acc = node_correct / max(node_total, 1)
    edge_acc = edge_correct / max(edge_total, 1)

    return {
        "val/loss": avg_val_loss,
        "val/node_accuracy": node_acc,
        "val/edge_accuracy": edge_acc,
    }


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train(cfg: DictConfig) -> None:
    """Run the MDLM training loop."""

    # ---- 1. Seed and device ----
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # ---- 2. Output directory ----
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = _PROJECT_ROOT / "outputs" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    # Save resolved config to output directory
    config_path = output_dir / "config.yaml"
    config_path.write_text(OmegaConf.to_yaml(cfg, resolve=True))

    # ---- 3. wandb ----
    init_wandb(cfg)

    # ---- 4. VocabConfig ----
    vocab_config = VocabConfig(n_max=cfg.data.n_max)

    # ---- 5. Datasets ----
    mat_path = _PROJECT_ROOT / cfg.data.mat_path
    cache_path = _PROJECT_ROOT / cfg.data.cache_path

    train_ds = BubbleDiagramDataset(
        mat_path=mat_path,
        cache_path=cache_path,
        vocab_config=vocab_config,
        split="train",
        train_frac=cfg.data.splits.train,
        val_frac=cfg.data.splits.val,
        test_frac=cfg.data.splits.test,
        seed=cfg.seed,
    )
    val_ds = BubbleDiagramDataset(
        mat_path=mat_path,
        cache_path=cache_path,
        vocab_config=vocab_config,
        split="val",
        train_frac=cfg.data.splits.train,
        val_frac=cfg.data.splits.val,
        test_frac=cfg.data.splits.test,
        seed=cfg.seed,
    )

    # ---- 6. DataLoaders ----
    num_workers = cfg.data.num_workers
    if platform.system() == "Windows" and num_workers > 0:
        logger.warning(
            "Windows detected: setting num_workers=0 to avoid "
            "multiprocessing issues (was %d)",
            num_workers,
        )
        num_workers = 0

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=(device == "cuda"),
    )

    # ---- 7. Model, noise schedule, loss, optimizer ----
    model = BDDenoiser(
        d_model=cfg.model.d_model,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        vocab_config=vocab_config,
        cond_dim=cfg.model.cond_dim,
        mlp_ratio=cfg.model.mlp_ratio,
        dropout=cfg.model.dropout,
        frequency_embedding_size=cfg.model.frequency_embedding_size,
    ).to(device)

    noise_schedule = get_noise(cfg.noise).to(device)

    criterion = ELBOLoss(
        edge_class_weights=train_ds.edge_class_weights,
        node_class_weights=None,
        vocab_config=vocab_config,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    # ---- 8. Learning-rate schedule ----
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _build_lr_lambda(cfg.training.warmup_steps),
    )

    # ---- 9. Log model info ----
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %d (%.2fM)", n_params, n_params / 1e6)
    logger.info("Training samples: %d", len(train_ds))
    logger.info("Validation samples: %d", len(val_ds))
    logger.info("Batch size: %d", cfg.data.batch_size)
    logger.info(
        "Batches/epoch: %d (drop_last=True)",
        len(train_ds) // cfg.data.batch_size,
    )

    # ---- 10. Training loop ----
    global_step = 0
    checkpoint_dir = output_dir / "checkpoints"

    for epoch in range(cfg.training.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            tokens = batch["tokens"].to(device)
            pad_mask = batch["pad_mask"].to(device)

            B = tokens.size(0)
            t = torch.rand(B, device=device)

            if cfg.training.importance_sampling:
                t = noise_schedule.importance_sampling_transformation(t)

            t = torch.clamp(t, min=1e-5, max=1.0)

            x_t, mask_indicators = forward_mask(
                tokens, pad_mask, t, noise_schedule, vocab_config,
            )

            node_logits, edge_logits = model(x_t, pad_mask, t)

            loss = criterion(
                node_logits, edge_logits,
                tokens, x_t, pad_mask, mask_indicators,
                t, noise_schedule,
            )

            optimizer.zero_grad()
            loss.backward()

            if cfg.training.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.training.grad_clip,
                )

            optimizer.step()
            scheduler.step()

            current_lr = scheduler.get_last_lr()[0]
            log_metrics(
                {"train/loss": loss.item(), "train/lr": current_lr},
                step=global_step,
            )

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        logger.info(
            "Epoch %d: train_loss=%.4f, lr=%.2e, steps=%d",
            epoch, avg_train_loss, current_lr, global_step,
        )

        # ---- 11. Validation ----
        if (epoch + 1) % cfg.training.val_every == 0:
            val_metrics = _validate(
                model, val_loader, criterion, noise_schedule,
                vocab_config, device,
            )
            log_metrics(val_metrics, step=global_step)
            logger.info(
                "  Validation: loss=%.4f, node_acc=%.3f, edge_acc=%.3f",
                val_metrics["val/loss"],
                val_metrics["val/node_accuracy"],
                val_metrics["val/edge_accuracy"],
            )

        # ---- 12. Sampling ----
        if (epoch + 1) % cfg.training.sample_every == 0:
            model.eval()
            with torch.no_grad():
                samples = sample(
                    model=model,
                    noise_schedule=noise_schedule,
                    vocab_config=vocab_config,
                    batch_size=8,
                    num_steps=50,
                    temperature=0.0,
                    num_rooms_distribution=train_ds.num_rooms_distribution,
                    device=device,
                )
            log_metrics(
                {"samples/num_generated": samples.size(0)},
                step=global_step,
            )
            logger.info(
                "  Generated %d samples at epoch %d", samples.size(0), epoch,
            )

        # ---- 13. Checkpoint ----
        if (epoch + 1) % cfg.training.checkpoint_every == 0:
            ckpt_path = checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt"
            save_checkpoint(model, optimizer, epoch, cfg, ckpt_path)

    # ---- 14. Final checkpoint and cleanup ----
    final_path = checkpoint_dir / "checkpoint_final.pt"
    save_checkpoint(model, optimizer, cfg.training.epochs - 1, cfg, final_path)

    wandb.finish()
    logger.info("Training complete. Total steps: %d", global_step)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Collect CLI overrides (everything after the script name)
    cli_overrides = sys.argv[1:]
    config = _load_config(cli_overrides)
    train(config)
