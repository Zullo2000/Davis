"""MDLM v2 training loop with learned forward process (MELD).

Jointly trains the BDDenoiser and RateNetwork using STGS (Straight-Through
Gumbel-Softmax) for gradient flow through the discrete masking step.

Uses Hydra's Compose API (same pattern as train.py).

Usage:
    python scripts/train_v2.py noise=learned training=v2
    python scripts/train_v2.py noise=learned training=v2 wandb.mode=disabled
    python scripts/train_v2.py noise=learned training=v2 training.epochs=2
"""

from __future__ import annotations

import logging
import math
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
from bd_gen.diffusion.forward_process import (  # noqa: E402
    forward_mask_eval_learned,
    forward_mask_learned,
)
from bd_gen.diffusion.loss import ELBOLossV2  # noqa: E402
from bd_gen.diffusion.rate_network import RateNetwork  # noqa: E402
from bd_gen.diffusion.sampling import sample  # noqa: E402
from bd_gen.model.denoiser import BDDenoiser  # noqa: E402
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
    """Load and compose the Hydra config with CLI overrides."""
    config_dir = str((_PROJECT_ROOT / "configs").resolve())
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config", overrides=overrides or [])
    return cfg


# ---------------------------------------------------------------------------
# Learning-rate schedule
# ---------------------------------------------------------------------------


def _build_lr_lambda(warmup_steps: int):
    """Return a lambda for LambdaLR implementing linear warmup then constant."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return 1.0

    return lr_lambda


# ---------------------------------------------------------------------------
# Gumbel temperature schedule
# ---------------------------------------------------------------------------


def _compute_gumbel_temperature(
    epoch: int,
    total_epochs: int,
    start: float,
    end: float,
    decay: str,
) -> float:
    """Compute Gumbel-Softmax temperature for the current epoch."""
    progress = epoch / max(total_epochs - 1, 1)
    if decay == "linear":
        return start + progress * (end - start)
    elif decay == "cosine":
        return end + 0.5 * (start - end) * (1 + math.cos(math.pi * progress))
    return start


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------


def save_checkpoint_v2(
    model: torch.nn.Module,
    rate_network: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: DictConfig,
    path: str | Path,
) -> None:
    """Save a v2 training checkpoint (denoiser + rate network)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "rate_network_state_dict": rate_network.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "config": OmegaConf.to_container(config, resolve=True),
    }
    torch.save(checkpoint, path)
    logger.info("v2 checkpoint saved: %s (epoch %d)", path, epoch)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train_v2(cfg: DictConfig) -> None:
    """Run the v2 MDLM training loop with learned forward process."""

    # ---- 1. Seed and device ----
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # ---- 2. Output directory ----
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = _PROJECT_ROOT / "outputs" / f"v2_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

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
            "Windows detected: setting num_workers=0 (was %d)", num_workers,
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

    # ---- 7. Models ----
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

    rate_network = RateNetwork(
        vocab_config=vocab_config,
        d_emb=cfg.noise.d_emb,
        K=cfg.noise.K,
        gamma_min=cfg.noise.gamma_min,
        gamma_max=cfg.noise.gamma_max,
        hidden_dim=cfg.noise.hidden_dim,
    ).to(device)

    # ---- 8. Single optimizer for both networks ----
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(rate_network.parameters()),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    # ---- 9. Loss ----
    criterion = ELBOLossV2(
        edge_class_weights=train_ds.edge_class_weights,
        node_class_weights=None,
        vocab_config=vocab_config,
        lambda_edge=cfg.training.lambda_edge,
    ).to(device)

    # ---- 10. Learning-rate schedule ----
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _build_lr_lambda(cfg.training.warmup_steps),
    )

    # ---- 11. Gumbel temperature config ----
    gumbel_start = cfg.training.gumbel_temperature_start
    gumbel_end = cfg.training.gumbel_temperature_end
    gumbel_decay = cfg.training.gumbel_temperature_decay

    # ---- 12. Log model info ----
    n_params_model = sum(p.numel() for p in model.parameters())
    n_params_rate = sum(p.numel() for p in rate_network.parameters())
    logger.info(
        "Denoiser parameters: %d (%.2fM)", n_params_model, n_params_model / 1e6,
    )
    logger.info(
        "Rate network parameters: %d (%.2fK)", n_params_rate, n_params_rate / 1e3,
    )
    logger.info("Training samples: %d", len(train_ds))
    logger.info("Validation samples: %d", len(val_ds))
    logger.info("Batch size: %d", cfg.data.batch_size)

    # ---- 13. Training loop ----
    global_step = 0
    checkpoint_dir = output_dir / "checkpoints"

    for epoch in range(cfg.training.epochs):
        model.train()
        rate_network.train()
        epoch_loss = 0.0
        n_batches = 0

        gumbel_temp = _compute_gumbel_temperature(
            epoch, cfg.training.epochs,
            gumbel_start, gumbel_end, gumbel_decay,
        )

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            tokens = batch["tokens"].to(device)
            pad_mask = batch["pad_mask"].to(device)

            B = tokens.size(0)

            # Uniform t (no importance sampling for v2)
            t = torch.rand(B, device=device)
            t = torch.clamp(t, min=1e-5, max=1.0)

            # Forward masking with STGS
            stgs_out = forward_mask_learned(
                tokens, pad_mask, t, rate_network, model,
                vocab_config, gumbel_temperature=gumbel_temp,
            )

            # Denoiser with soft embeddings
            node_logits, edge_logits = model(
                tokens, pad_mask, t,
                pre_embedded=stgs_out["soft_embeddings"],
            )

            # Rate network derivatives for loss weights
            rate_out = rate_network.forward_with_derivative(t, pad_mask)

            # Per-position ELBO loss
            loss = criterion(
                node_logits, edge_logits, tokens,
                pad_mask, stgs_out["mask_indicators"],
                rate_out["alpha"], rate_out["alpha_prime"],
            )

            optimizer.zero_grad()
            loss.backward()

            if cfg.training.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(rate_network.parameters()),
                    cfg.training.grad_clip,
                )

            optimizer.step()
            scheduler.step()

            current_lr = scheduler.get_last_lr()[0]
            log_metrics(
                {
                    "train/loss": loss.item(),
                    "train/lr": current_lr,
                    "train/gumbel_temp": gumbel_temp,
                },
                step=global_step,
            )

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        logger.info(
            "Epoch %d: train_loss=%.4f, lr=%.2e, gumbel_temp=%.3f",
            epoch, avg_train_loss, current_lr, gumbel_temp,
        )

        # ---- 14. Validation (discrete masking, no STGS) ----
        if (epoch + 1) % cfg.training.val_every == 0:
            model.eval()
            rate_network.eval()
            val_loss_sum = 0.0
            val_batches = 0

            with torch.no_grad():
                for batch in val_loader:
                    tokens = batch["tokens"].to(device)
                    pad_mask = batch["pad_mask"].to(device)

                    B = tokens.size(0)
                    t = torch.rand(B, device=device)
                    t = torch.clamp(t, min=1e-5, max=1.0)

                    # Eval path: discrete masking, no STGS
                    x_t, mask_ind = forward_mask_eval_learned(
                        tokens, pad_mask, t, rate_network, vocab_config,
                    )
                    # pre_embedded=None → v1 embedding path
                    node_logits, edge_logits = model(x_t, pad_mask, t)
                    rate_out = rate_network.forward_with_derivative(t, pad_mask)
                    loss = criterion(
                        node_logits, edge_logits, tokens,
                        pad_mask, mask_ind,
                        rate_out["alpha"], rate_out["alpha_prime"],
                    )

                    val_loss_sum += loss.item()
                    val_batches += 1

            avg_val_loss = val_loss_sum / max(val_batches, 1)
            log_metrics({"val/loss": avg_val_loss}, step=global_step)
            logger.info("  Validation: loss=%.4f", avg_val_loss)

        # ---- 15. Sampling ----
        if (epoch + 1) % cfg.training.sample_every == 0:
            model.eval()
            rate_network.eval()
            with torch.no_grad():
                # Dummy noise schedule for sample() signature compatibility
                from bd_gen.diffusion.noise_schedule import LogLinearSchedule

                dummy_schedule = LogLinearSchedule().to(device)
                samples = sample(
                    model=model,
                    noise_schedule=dummy_schedule,
                    vocab_config=vocab_config,
                    batch_size=8,
                    num_steps=50,
                    temperature=0.0,
                    rate_network=rate_network,
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

        # ---- 16. Checkpoint ----
        if (epoch + 1) % cfg.training.checkpoint_every == 0:
            ckpt_path = checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt"
            save_checkpoint_v2(
                model, rate_network, optimizer, epoch, cfg, ckpt_path,
            )

    # ---- 17. Final checkpoint and cleanup ----
    final_path = checkpoint_dir / "checkpoint_final.pt"
    save_checkpoint_v2(
        model, rate_network, optimizer,
        cfg.training.epochs - 1, cfg, final_path,
    )

    wandb.finish()
    logger.info("Training complete. Total steps: %d", global_step)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli_overrides = sys.argv[1:]
    config = _load_config(cli_overrides)
    train_v2(config)
