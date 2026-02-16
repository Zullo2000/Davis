"""Full evaluation pipeline: generate samples, compute metrics, log results.

Uses Hydra's Compose API (same pattern as train.py).

Usage::

    python scripts/evaluate.py eval.checkpoint_path=path/to/ckpt.pt
    python scripts/evaluate.py eval.checkpoint_path=... eval.num_samples=500
    python scripts/evaluate.py eval.checkpoint_path=... wandb.mode=disabled
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch
import wandb
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Ensure BD_Generation is on sys.path when running as a script
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from bd_gen.data.dataset import BubbleDiagramDataset  # noqa: E402
from bd_gen.data.tokenizer import detokenize  # noqa: E402
from bd_gen.data.vocab import NODE_PAD_IDX, VocabConfig  # noqa: E402
from bd_gen.diffusion.noise_schedule import get_noise  # noqa: E402
from bd_gen.diffusion.sampling import sample  # noqa: E402
from bd_gen.eval.metrics import (  # noqa: E402
    distribution_match,
    diversity,
    novelty,
    validity_rate,
)
from bd_gen.eval.validity import check_validity_batch  # noqa: E402
from bd_gen.model.denoiser import BDDenoiser  # noqa: E402
from bd_gen.utils.checkpoint import load_checkpoint  # noqa: E402
from bd_gen.utils.logging_utils import init_wandb, log_metrics  # noqa: E402
from bd_gen.utils.seed import set_seed  # noqa: E402
from bd_gen.viz.graph_viz import draw_bubble_diagram_grid  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _load_config(overrides: list[str] | None = None) -> DictConfig:
    """Load and compose the Hydra config with CLI overrides."""
    config_dir = str((_PROJECT_ROOT / "configs").resolve())
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config", overrides=overrides or [])
    return cfg


def _reconstruct_pad_masks(
    tokens: torch.Tensor,
    vocab_config: VocabConfig,
) -> torch.Tensor:
    """Reconstruct pad masks from generated tokens.

    Counts non-PAD nodes to infer num_rooms, then computes the
    canonical pad mask.
    """
    pad_masks = torch.zeros_like(tokens, dtype=torch.bool)
    for i in range(tokens.size(0)):
        node_tokens = tokens[i, :vocab_config.n_max]
        num_rooms = int((node_tokens != NODE_PAD_IDX).sum().item())
        num_rooms = max(1, min(num_rooms, vocab_config.n_max))
        pad_masks[i] = vocab_config.compute_pad_mask(num_rooms)
    return pad_masks


def evaluate(cfg: DictConfig) -> None:
    """Run the full evaluation pipeline."""
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # --- Output directory ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = _PROJECT_ROOT / "outputs" / f"eval_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))

    # --- wandb ---
    init_wandb(cfg)

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

    load_checkpoint(ckpt_path, model, optimizer=None, device=device)
    model.eval()
    logger.info("Model loaded from %s", ckpt_path)

    # --- Noise schedule ---
    noise_schedule = get_noise(cfg.noise).to(device)

    # --- Load dataset for reference statistics ---
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

    # --- Generate samples in batches ---
    num_samples = cfg.eval.num_samples
    batch_size = cfg.eval.batch_size
    logger.info(
        "Generating %d samples (steps=%d, temp=%.2f, batch=%d)",
        num_samples, cfg.eval.sampling_steps, cfg.eval.temperature, batch_size,
    )

    all_tokens = []
    with torch.no_grad():
        for start in tqdm(range(0, num_samples, batch_size), desc="Generating"):
            bs = min(batch_size, num_samples - start)
            batch_tokens = sample(
                model=model,
                noise_schedule=noise_schedule,
                vocab_config=vocab_config,
                batch_size=bs,
                num_steps=cfg.eval.sampling_steps,
                temperature=cfg.eval.temperature,
                num_rooms_distribution=num_rooms_dist,
                device=device,
            )
            all_tokens.append(batch_tokens.cpu())

    tokens = torch.cat(all_tokens, dim=0)
    pad_masks = _reconstruct_pad_masks(tokens, vocab_config)
    logger.info("Generated %d samples", tokens.size(0))

    # --- Validity check ---
    logger.info("Running validity checks...")
    validity_results = check_validity_batch(tokens, pad_masks, vocab_config)
    v_rate = validity_rate(validity_results)
    logger.info("Validity rate: %.1f%% (%d / %d)",
                100 * v_rate, sum(1 for r in validity_results if r["overall"]),
                len(validity_results))

    # --- Detokenize samples ---
    graph_dicts = []
    for i in range(tokens.size(0)):
        try:
            gd = detokenize(tokens[i], pad_masks[i], vocab_config)
            graph_dicts.append(gd)
        except ValueError:
            graph_dicts.append({"num_rooms": 0, "node_types": [], "edge_triples": []})

    # --- Detokenize training set for comparison ---
    logger.info("Detokenizing training set for comparison metrics...")
    train_dicts = []
    for idx in range(len(train_ds)):
        item = train_ds[idx]
        try:
            gd = detokenize(item["tokens"], item["pad_mask"], vocab_config)
            train_dicts.append(gd)
        except ValueError:
            pass

    # --- Compute metrics ---
    metrics_dict: dict[str, float] = {"eval/validity_rate": v_rate}

    requested = cfg.eval.metrics

    if "diversity" in requested:
        div = diversity(graph_dicts)
        metrics_dict["eval/diversity"] = div
        logger.info("Diversity: %.3f", div)

    if "novelty" in requested:
        nov = novelty(graph_dicts, train_dicts)
        metrics_dict["eval/novelty"] = nov
        logger.info("Novelty: %.3f", nov)

    if "distribution_match" in requested:
        dm = distribution_match(graph_dicts, train_dicts)
        metrics_dict["eval/node_kl"] = dm["node_kl"]
        metrics_dict["eval/edge_kl"] = dm["edge_kl"]
        metrics_dict["eval/num_rooms_kl"] = dm["num_rooms_kl"]
        logger.info(
            "Distribution match — node=%.4f, edge=%.4f, rooms=%.4f",
            dm["node_kl"], dm["edge_kl"], dm["num_rooms_kl"],
        )

    # --- Detailed validity breakdown ---
    n_res = len(validity_results)
    connected_rate = sum(
        1 for r in validity_results if r["connected"]
    ) / n_res
    valid_types_rate = sum(
        1 for r in validity_results if r["valid_types"]
    ) / n_res
    no_mask_rate = sum(
        1 for r in validity_results if r["no_mask_tokens"]
    ) / n_res
    metrics_dict["eval/connected_rate"] = connected_rate
    metrics_dict["eval/valid_types_rate"] = valid_types_rate
    metrics_dict["eval/no_mask_rate"] = no_mask_rate

    logger.info(
        "Breakdown — connected=%.1f%%, types=%.1f%%, no_mask=%.1f%%",
        100 * connected_rate, 100 * valid_types_rate,
        100 * no_mask_rate,
    )

    # --- Log to wandb ---
    log_metrics(metrics_dict)

    # --- Save results ---
    results_path = output_dir / "metrics.json"
    results_path.write_text(json.dumps(metrics_dict, indent=2))
    logger.info("Metrics saved: %s", results_path)

    # --- Save samples ---
    if cfg.eval.save_samples:
        torch.save({"tokens": tokens, "pad_masks": pad_masks},
                    output_dir / "samples.pt")

    # --- Visualize ---
    if cfg.eval.visualize:
        n_viz = min(cfg.eval.num_viz_samples, len(graph_dicts))
        viz_dicts = [g for g in graph_dicts[:n_viz] if g["num_rooms"] > 0]
        if viz_dicts:
            fig = draw_bubble_diagram_grid(viz_dicts)
            fig_path = output_dir / "samples.png"
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            logger.info("Visualization saved: %s", fig_path)

            # Log to wandb
            try:
                wandb.log({"eval/samples": wandb.Image(str(fig_path))})
            except Exception:
                pass  # wandb may be disabled

            import matplotlib.pyplot as plt
            plt.close(fig)

    wandb.finish()
    logger.info("Evaluation complete. Output: %s", output_dir)


if __name__ == "__main__":
    cli_overrides = sys.argv[1:]
    config = _load_config(cli_overrides)
    evaluate(config)
