"""Generate and visualize bubble diagram samples from a trained checkpoint.

Uses Hydra's Compose API (same pattern as train.py).

Usage::

    python scripts/sample.py eval.checkpoint_path=path/to/ckpt.pt
    python scripts/sample.py eval.checkpoint_path=... eval.num_samples=8
    python scripts/sample.py eval.checkpoint_path=... eval.temperature=0.5
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

# Ensure BD_Generation is on sys.path when running as a script
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from bd_gen.data.dataset import BubbleDiagramDataset  # noqa: E402
from bd_gen.data.tokenizer import detokenize  # noqa: E402
from bd_gen.data.vocab import VocabConfig  # noqa: E402
from bd_gen.diffusion.noise_schedule import get_noise  # noqa: E402
from bd_gen.diffusion.sampling import sample  # noqa: E402
from bd_gen.eval.validity import check_validity  # noqa: E402
from bd_gen.model.denoiser import BDDenoiser  # noqa: E402
from bd_gen.utils.checkpoint import load_checkpoint  # noqa: E402
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


def generate(cfg: DictConfig) -> None:
    """Generate samples from a trained checkpoint."""
    set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # --- Output directory ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = _PROJECT_ROOT / "outputs" / f"sample_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config
    (output_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))

    # --- Check checkpoint path ---
    ckpt_path = cfg.eval.checkpoint_path
    if ckpt_path is None:
        logger.error("eval.checkpoint_path is required. Set via CLI override.")
        sys.exit(1)
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_absolute():
        ckpt_path = _PROJECT_ROOT / ckpt_path
    logger.info("Checkpoint: %s", ckpt_path)

    # --- Build model ---
    vocab_config = VocabConfig(n_max=cfg.data.n_max)
    model = BDDenoiser(
        d_model=cfg.model.d_model,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        vocab_config=vocab_config,
        cond_dim=cfg.model.cond_dim,
        mlp_ratio=cfg.model.mlp_ratio,
        dropout=0.0,  # No dropout at inference
        frequency_embedding_size=cfg.model.frequency_embedding_size,
    ).to(device)

    load_checkpoint(ckpt_path, model, optimizer=None, device=device)
    model.eval()

    # --- Noise schedule ---
    noise_schedule = get_noise(cfg.noise).to(device)

    # --- Num rooms distribution (optional) ---
    num_rooms_dist = None
    try:
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
        logger.info("Loaded num_rooms_distribution from training set")
    except Exception as exc:
        logger.warning("Could not load dataset for num_rooms_distribution: %s", exc)

    # --- Generate samples ---
    num_samples = cfg.eval.num_samples
    logger.info("Generating %d samples (steps=%d, temp=%.2f)",
                num_samples, cfg.eval.sampling_steps, cfg.eval.temperature)

    with torch.no_grad():
        tokens = sample(
            model=model,
            noise_schedule=noise_schedule,
            vocab_config=vocab_config,
            batch_size=num_samples,
            num_steps=cfg.eval.sampling_steps,
            temperature=cfg.eval.temperature,
            num_rooms_distribution=num_rooms_dist,
            device=device,
        )

    # --- Compute pad masks ---
    pad_masks = torch.zeros_like(tokens, dtype=torch.bool)
    for i in range(tokens.size(0)):
        # Count real nodes (non-PAD in node positions)
        from bd_gen.data.vocab import NODE_PAD_IDX
        node_tokens = tokens[i, :vocab_config.n_max]
        num_rooms = int((node_tokens != NODE_PAD_IDX).sum().item())
        num_rooms = max(1, min(num_rooms, vocab_config.n_max))
        pad_masks[i] = vocab_config.compute_pad_mask(num_rooms)

    # --- Detokenize and check validity ---
    graph_dicts = []
    valid_count = 0
    for i in range(tokens.size(0)):
        try:
            gd = detokenize(tokens[i], pad_masks[i], vocab_config)
            graph_dicts.append(gd)
            result = check_validity(tokens[i], pad_masks[i], vocab_config)
            if result["overall"]:
                valid_count += 1
        except ValueError as exc:
            logger.warning("Sample %d detokenize failed: %s", i, exc)
            graph_dicts.append({"num_rooms": 0, "node_types": [], "edge_triples": []})

    logger.info("Valid samples: %d / %d (%.1f%%)",
                valid_count, num_samples, 100 * valid_count / max(num_samples, 1))

    # --- Save tokens ---
    torch.save({"tokens": tokens.cpu(), "pad_masks": pad_masks.cpu()},
               output_dir / "samples.pt")

    # --- Save graph dicts as JSON ---
    serializable = [
        {
            "num_rooms": g["num_rooms"],
            "node_types": g["node_types"],
            "edge_triples": [list(t) for t in g["edge_triples"]],
        }
        for g in graph_dicts
    ]
    (output_dir / "samples.json").write_text(json.dumps(serializable, indent=2))

    # --- Visualize ---
    if cfg.eval.visualize:
        n_viz = min(cfg.eval.num_viz_samples, len(graph_dicts))
        viz_dicts = [g for g in graph_dicts[:n_viz] if g["num_rooms"] > 0]
        if viz_dicts:
            fig = draw_bubble_diagram_grid(viz_dicts)
            fig_path = output_dir / "samples.png"
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            logger.info("Visualization saved: %s", fig_path)
            import matplotlib.pyplot as plt
            plt.close(fig)

    logger.info("Output directory: %s", output_dir)


if __name__ == "__main__":
    cli_overrides = sys.argv[1:]
    config = _load_config(cli_overrides)
    generate(config)
