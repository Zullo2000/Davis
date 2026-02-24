"""Generate and save sample tokens for later metric evaluation (GPU).

Loads model checkpoint, generates tokens for each seed, and saves them
to ``eval_results/{schedule}/{method}_samples.pt``.  No metrics are
computed — use ``evaluate.py`` for that.

Usage::

    python scripts/generate_samples.py eval.checkpoint_path=path/to/ckpt.pt
    python scripts/generate_samples.py eval.checkpoint_path=... eval.num_samples=500
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from tqdm import tqdm

# Ensure BD_Generation is on sys.path when running as a script
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from bd_gen.data.dataset import BubbleDiagramDataset  # noqa: E402
from bd_gen.data.vocab import NODE_PAD_IDX, VocabConfig  # noqa: E402
from bd_gen.diffusion.noise_schedule import LogLinearSchedule, get_noise  # noqa: E402
from bd_gen.diffusion.rate_network import RateNetwork  # noqa: E402
from bd_gen.diffusion.remasking import create_remasking_schedule  # noqa: E402
from bd_gen.diffusion.sampling import sample  # noqa: E402
from bd_gen.model.denoiser import BDDenoiser  # noqa: E402
from bd_gen.utils.checkpoint import load_checkpoint  # noqa: E402
from bd_gen.utils.seed import set_seed  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (extracted from generate_and_evaluate.py)
# ---------------------------------------------------------------------------


def _load_config(overrides: list[str] | None = None) -> DictConfig:
    config_dir = str((_PROJECT_ROOT / "configs").resolve())
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config", overrides=overrides or [])
    return cfg


def _reconstruct_pad_masks(
    tokens: torch.Tensor,
    vocab_config: VocabConfig,
) -> torch.Tensor:
    pad_masks = torch.zeros_like(tokens, dtype=torch.bool)
    for i in range(tokens.size(0)):
        node_tokens = tokens[i, : vocab_config.n_max]
        num_rooms = int((node_tokens != NODE_PAD_IDX).sum().item())
        num_rooms = max(1, min(num_rooms, vocab_config.n_max))
        pad_masks[i] = vocab_config.compute_pad_mask(num_rooms)
    return pad_masks


def _load_v2_checkpoint(
    ckpt_path: Path,
    model: torch.nn.Module,
    vocab_config: VocabConfig,
    cfg: DictConfig,
    device: str,
) -> RateNetwork | None:
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    if "rate_network_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("v2 denoiser loaded from %s", ckpt_path)

        ckpt_cfg = checkpoint.get("config", {})
        noise_cfg = ckpt_cfg.get("noise", {})
        rate_network = RateNetwork(
            vocab_config=vocab_config,
            d_emb=noise_cfg.get("d_emb", cfg.noise.get("d_emb", 32)),
            K=noise_cfg.get("K", cfg.noise.get("K", 4)),
            gamma_min=noise_cfg.get("gamma_min", cfg.noise.get("gamma_min", -13.0)),
            gamma_max=noise_cfg.get("gamma_max", cfg.noise.get("gamma_max", 5.0)),
            hidden_dim=noise_cfg.get("hidden_dim", cfg.noise.get("hidden_dim", 64)),
        ).to(device)
        rate_network.load_state_dict(checkpoint["rate_network_state_dict"])
        rate_network.eval()
        n_params = sum(p.numel() for p in rate_network.parameters())
        logger.info("v2 rate network loaded (%d params)", n_params)
        return rate_network
    else:
        load_checkpoint(ckpt_path, model, optimizer=None, device=device)
        return None


def _build_method_name(cfg: DictConfig, is_v2: bool) -> str:
    """Build systematic method name: {unmasking}_{sampling}_{remasking}."""
    remasking_cfg = cfg.eval.remasking
    unmasking_mode = cfg.eval.get("unmasking_mode", "random")
    top_p = cfg.eval.get("top_p", None)

    unmask_tag = unmasking_mode
    if top_p is not None and top_p < 1.0:
        sampling_tag = f"topp{top_p}"
    elif cfg.eval.temperature == 0.0:
        sampling_tag = "argmax"
    else:
        sampling_tag = f"temp{cfg.eval.temperature}"

    if remasking_cfg.enabled:
        if remasking_cfg.strategy == "confidence":
            t_sw = remasking_cfg.get("t_switch", 1.0)
            remask_tag = f"remdm_confidence_tsw{t_sw}"
        else:
            remask_tag = (
                f"remdm_{remasking_cfg.strategy}_eta{remasking_cfg.eta}"
            )
    else:
        remask_tag = "no_remask"

    method_name = f"{unmask_tag}_{sampling_tag}_{remask_tag}"
    if is_v2:
        method_name = f"v2_{method_name}"
    return method_name


def _build_config_dict(cfg: DictConfig, ckpt_path: Path, is_v2: bool) -> dict:
    """Build config metadata dict stored alongside samples."""
    remasking_cfg = cfg.eval.remasking
    top_p = cfg.eval.get("top_p", None)
    return {
        "seeds": list(cfg.eval.get("seeds", [cfg.seed])),
        "num_samples": cfg.eval.num_samples,
        "sampling_steps": cfg.eval.sampling_steps,
        "temperature": cfg.eval.temperature,
        "top_p": top_p,
        "unmasking_mode": cfg.eval.get("unmasking_mode", "random"),
        "remasking_enabled": remasking_cfg.enabled,
        "remasking_strategy": (
            remasking_cfg.strategy if remasking_cfg.enabled else None
        ),
        "remasking_eta": (
            remasking_cfg.eta if remasking_cfg.enabled else None
        ),
        "remasking_t_switch": (
            remasking_cfg.get("t_switch", 1.0) if remasking_cfg.enabled
            else None
        ),
        "checkpoint": str(ckpt_path.name),
        "is_v2": is_v2,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def generate_samples(cfg: DictConfig) -> None:
    """Generate and save sample tokens for all seeds."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # --- Check checkpoint path ---
    ckpt_path = cfg.eval.checkpoint_path
    if ckpt_path is None:
        logger.error("eval.checkpoint_path is required. Set via CLI override.")
        sys.exit(1)
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_absolute():
        ckpt_path = _PROJECT_ROOT / ckpt_path

    # --- Build model ---
    vocab_config = VocabConfig(n_max=cfg.data.n_max)
    model = BDDenoiser(
        d_model=cfg.model.d_model,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        vocab_config=vocab_config,
        cond_dim=cfg.model.cond_dim,
        mlp_ratio=cfg.model.mlp_ratio,
        dropout=0.0,
        frequency_embedding_size=cfg.model.frequency_embedding_size,
    ).to(device)

    rate_network = _load_v2_checkpoint(ckpt_path, model, vocab_config, cfg, device)
    is_v2 = rate_network is not None
    model.eval()
    logger.info("Model loaded from %s (v2=%s)", ckpt_path, is_v2)

    # --- Noise schedule ---
    if is_v2:
        noise_schedule = LogLinearSchedule().to(device)
        logger.info("v2 mode: dummy LogLinearSchedule (rate_network provides alpha)")
    else:
        noise_schedule = get_noise(cfg.noise).to(device)

    # --- Remasking schedule ---
    remasking_fn = create_remasking_schedule(
        cfg.eval.remasking, noise_schedule, vocab_config,
    )
    if remasking_fn is not None and is_v2:
        logger.warning(
            "Remasking not supported with v2 learned rates; disabling."
        )
        remasking_fn = None
    elif remasking_fn is not None:
        logger.info(
            "Remasking enabled: strategy=%s, eta=%.3f, t_switch=%.2f",
            cfg.eval.remasking.strategy,
            cfg.eval.remasking.eta,
            cfg.eval.remasking.get("t_switch", 1.0),
        )

    # --- num_rooms distribution from training set ---
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
    num_rooms_dist = train_ds.num_rooms_distribution

    # --- Generate samples for each seed ---
    seeds = list(cfg.eval.get("seeds", [cfg.seed]))
    num_samples = cfg.eval.num_samples
    batch_size = cfg.eval.batch_size
    unmasking_mode = cfg.eval.get("unmasking_mode", "random")

    logger.info(
        "Generating %d samples x %d seeds (%s)",
        num_samples, len(seeds), ", ".join(str(s) for s in seeds),
    )

    per_seed_data: dict[str, dict[str, torch.Tensor]] = {}
    for seed in seeds:
        logger.info("--- Seed %d ---", seed)
        set_seed(seed)

        all_tokens = []
        with torch.no_grad():
            for start in tqdm(
                range(0, num_samples, batch_size),
                desc=f"seed {seed}",
                leave=False,
            ):
                bs = min(batch_size, num_samples - start)
                batch_tokens = sample(
                    model=model,
                    noise_schedule=noise_schedule,
                    vocab_config=vocab_config,
                    batch_size=bs,
                    num_steps=cfg.eval.sampling_steps,
                    temperature=cfg.eval.temperature,
                    top_p=cfg.eval.get("top_p", None),
                    unmasking_mode=unmasking_mode,
                    t_switch=cfg.eval.remasking.get("t_switch", 1.0),
                    remasking_fn=remasking_fn,
                    rate_network=rate_network,
                    num_rooms_distribution=num_rooms_dist,
                    device=device,
                )
                all_tokens.append(batch_tokens.cpu())

        tokens = torch.cat(all_tokens, dim=0)
        pad_masks = _reconstruct_pad_masks(tokens, vocab_config)
        logger.info(
            "  Generated %d samples, shape %s",
            tokens.size(0), list(tokens.shape),
        )

        per_seed_data[str(seed)] = {"tokens": tokens, "pad_masks": pad_masks}

    # --- Save ---
    method_name = _build_method_name(cfg, is_v2)
    schedule_tag = cfg.noise.type
    eval_results_dir = _PROJECT_ROOT / "eval_results" / schedule_tag
    eval_results_dir.mkdir(parents=True, exist_ok=True)

    samples_path = eval_results_dir / f"{method_name}_samples.pt"
    payload = {
        "format_version": 1,
        "method": method_name,
        "n_max": cfg.data.n_max,
        "seeds": seeds,
        "num_samples": num_samples,
        "config": _build_config_dict(cfg, ckpt_path, is_v2),
        "per_seed": per_seed_data,
    }
    torch.save(payload, samples_path)
    size_mb = samples_path.stat().st_size / 1e6
    logger.info("Saved: %s (%.1f MB)", samples_path, size_mb)
    logger.info("Method name: %s", method_name)
    logger.info(
        "Run 'python scripts/evaluate.py --schedule %s"
        " --model %s' to compute metrics.",
        schedule_tag, method_name,
    )


if __name__ == "__main__":
    cli_overrides = sys.argv[1:]
    config = _load_config(cli_overrides)
    generate_samples(config)
