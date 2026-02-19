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

import numpy as np
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
from bd_gen.diffusion.remasking import create_remasking_schedule  # noqa: E402
from bd_gen.diffusion.sampling import sample  # noqa: E402
from bd_gen.eval.metrics import (  # noqa: E402
    conditional_edge_distances_topN,
    conditional_edge_kl,
    distribution_match,
    diversity,
    edge_present_rate_by_num_rooms,
    graph_structure_mmd,
    mode_coverage,
    novelty,
    spatial_transitivity,
    spatial_transitivity_by_num_rooms,
    type_conditioned_degree_kl,
    validity_by_num_rooms,
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
    """Reconstruct pad masks from generated tokens."""
    pad_masks = torch.zeros_like(tokens, dtype=torch.bool)
    for i in range(tokens.size(0)):
        node_tokens = tokens[i, :vocab_config.n_max]
        num_rooms = int((node_tokens != NODE_PAD_IDX).sum().item())
        num_rooms = max(1, min(num_rooms, vocab_config.n_max))
        pad_masks[i] = vocab_config.compute_pad_mask(num_rooms)
    return pad_masks


def _generate_and_evaluate_single_seed(
    cfg: DictConfig,
    model: torch.nn.Module,
    noise_schedule,
    vocab_config: VocabConfig,
    train_dicts: list[dict],
    num_rooms_dist,
    remasking_fn,
    device: str,
    seed: int,
) -> dict:
    """Generate samples with a given seed and compute all metrics.

    Returns a flat dict of metric_name -> float (or nested dicts for
    stratified metrics).
    """
    set_seed(seed)

    num_samples = cfg.eval.num_samples
    batch_size = cfg.eval.batch_size
    unmasking_mode = cfg.eval.get("unmasking_mode", "random")

    # --- Generate samples ---
    all_tokens = []
    with torch.no_grad():
        for start in range(0, num_samples, batch_size):
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
                num_rooms_distribution=num_rooms_dist,
                device=device,
            )
            all_tokens.append(batch_tokens.cpu())

    tokens = torch.cat(all_tokens, dim=0)
    pad_masks = _reconstruct_pad_masks(tokens, vocab_config)

    # --- Validity check ---
    validity_results = check_validity_batch(tokens, pad_masks, vocab_config)
    v_rate = validity_rate(validity_results)

    # --- Detokenize ---
    graph_dicts = []
    for i in range(tokens.size(0)):
        try:
            gd = detokenize(tokens[i], pad_masks[i], vocab_config)
            graph_dicts.append(gd)
        except ValueError:
            graph_dicts.append({"num_rooms": 0, "node_types": [], "edge_triples": []})

    # --- Compute metrics ---
    metrics: dict = {"eval/validity_rate": v_rate}
    requested = cfg.eval.metrics

    if "diversity" in requested:
        metrics["eval/diversity"] = diversity(graph_dicts)

    if "novelty" in requested:
        metrics["eval/novelty"] = novelty(graph_dicts, train_dicts)

    if "distribution_match" in requested:
        dm = distribution_match(graph_dicts, train_dicts)
        for key, val in dm.items():
            metrics[f"eval/{key}"] = val

    if "conditional_edge_kl" in requested:
        cekl = conditional_edge_kl(graph_dicts, train_dicts)
        for key, val in cekl.items():
            metrics[f"eval/{key}"] = val

    # Top-N conditional distances
    top_n = cfg.eval.get("conditional_topN_pairs", None)
    if top_n is not None and "conditional_edge_kl" in requested:
        topn_result = conditional_edge_distances_topN(
            graph_dicts, train_dicts, top_n=top_n,
        )
        for key, val in topn_result.items():
            metrics[f"eval/{key}"] = val

    if "graph_structure_mmd" in requested:
        mmd = graph_structure_mmd(
            graph_dicts, train_dicts, n_max=cfg.data.n_max,
        )
        for key, val in mmd.items():
            metrics[f"eval/{key}"] = val

    if "spatial_transitivity" in requested:
        st = spatial_transitivity(graph_dicts)
        for key, val in st.items():
            metrics[f"eval/{key}"] = val

    if "type_conditioned_degree_kl" in requested:
        tcdkl = type_conditioned_degree_kl(
            graph_dicts, train_dicts, n_max=cfg.data.n_max,
        )
        for key, val in tcdkl.items():
            metrics[f"eval/{key}"] = val

    if "mode_coverage" in requested:
        mc = mode_coverage(graph_dicts, train_dicts)
        for key, val in mc.items():
            metrics[f"eval/{key}"] = val

    # --- Detailed validity breakdown ---
    n_res = len(validity_results)
    metrics["eval/connected_rate"] = sum(
        1 for r in validity_results if r["connected"]
    ) / n_res
    metrics["eval/valid_types_rate"] = sum(
        1 for r in validity_results if r["valid_types"]
    ) / n_res
    metrics["eval/no_mask_rate"] = sum(
        1 for r in validity_results if r["no_mask_tokens"]
    ) / n_res

    # --- Stratified metrics ---
    if cfg.eval.get("stratified", True):
        vbn = validity_by_num_rooms(validity_results, graph_dicts)
        metrics["validity_by_num_rooms"] = vbn

        stbn = spatial_transitivity_by_num_rooms(graph_dicts)
        metrics["transitivity_by_num_rooms"] = stbn

        ebn = edge_present_rate_by_num_rooms(graph_dicts)
        metrics["edge_present_rate_by_num_rooms"] = ebn

    return metrics


def _aggregate_multi_seed(
    per_seed_results: dict[int, dict],
) -> dict[str, dict[str, float]]:
    """Aggregate scalar metrics across seeds into mean/std.

    Nested dicts (stratified metrics) are flattened with ``/`` separators
    and aggregated leaf-by-leaf.

    Returns:
        Dict mapping metric_name to {"mean": ..., "std": ...}.
    """

    def _flatten(d: dict, prefix: str = "") -> dict[str, float]:
        """Flatten nested dict into dot-separated keys, keeping only floats."""
        flat: dict[str, float] = {}
        for k, v in d.items():
            key = f"{prefix}/{k}" if prefix else k
            if isinstance(v, (int, float)):
                flat[key] = float(v)
            elif isinstance(v, dict):
                flat.update(_flatten(v, key))
        return flat

    # Flatten each seed's metrics
    all_flat: dict[int, dict[str, float]] = {}
    for seed, metrics in per_seed_results.items():
        all_flat[seed] = _flatten(metrics)

    # Collect all keys across seeds
    all_keys: set[str] = set()
    for flat in all_flat.values():
        all_keys.update(flat.keys())

    # Compute mean/std per key
    summary: dict[str, dict[str, float]] = {}
    for key in sorted(all_keys):
        values = [flat[key] for flat in all_flat.values() if key in flat]
        if values:
            arr = np.array(values, dtype=np.float64)
            summary[key] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=0)),
            }

    return summary


def _prefix_metrics(metrics: dict[str, float]) -> dict[str, float]:
    """Add scoreboard prefixes to flat metric keys for wandb grouping."""
    prefix_map = {
        "eval/validity_rate": "sampler/validity/overall",
        "eval/connected_rate": "sampler/validity/connected",
        "eval/valid_types_rate": "sampler/validity/valid_types",
        "eval/no_mask_rate": "sampler/validity/no_mask_tokens",
        "eval/diversity": "sampler/coverage/diversity",
        "eval/novelty": "sampler/coverage/novelty",
        "eval/mode_coverage": "sampler/coverage/mode_coverage",
        "eval/mode_coverage_weighted": "sampler/coverage/mode_coverage_weighted",
        "eval/node_kl": "sampler/distribution/node_kl",
        "eval/edge_kl": "sampler/distribution/edge_kl",
        "eval/num_rooms_kl": "sampler/distribution/rooms_kl",
        "eval/node_js": "sampler/distribution/node_js",
        "eval/edge_js": "sampler/distribution/edge_js",
        "eval/node_tv": "sampler/distribution/node_tv",
        "eval/edge_tv": "sampler/distribution/edge_tv",
        "eval/rooms_w1": "sampler/distribution/rooms_w1",
        "eval/mmd_degree": "sampler/structure/mmd_degree",
        "eval/mmd_clustering": "sampler/structure/mmd_clustering",
        "eval/mmd_spectral": "sampler/structure/mmd_spectral",
        "eval/transitivity_score": "sampler/structure/transitivity_score",
        "eval/h_consistent": "sampler/structure/h_consistent",
        "eval/v_consistent": "sampler/structure/v_consistent",
    }
    prefixed = {}
    for key, val in metrics.items():
        if isinstance(val, (int, float)):
            if key in prefix_map:
                prefixed[prefix_map[key]] = val
            # Also keep conditional/degree metrics with sampler/ prefix
            elif key.startswith("eval/conditional_edge"):
                prefixed[key.replace("eval/", "sampler/conditional/")] = val
            elif key.startswith("eval/degree_"):
                prefixed[key.replace("eval/", "sampler/conditional/")] = val
    return prefixed


def evaluate(cfg: DictConfig) -> None:
    """Run the full evaluation pipeline."""
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

    # --- Remasking schedule (ReMDM) ---
    remasking_fn = create_remasking_schedule(
        cfg.eval.remasking, noise_schedule, vocab_config,
    )
    if remasking_fn is not None:
        logger.info(
            "Remasking enabled: strategy=%s, eta=%.3f, t_switch=%.2f",
            cfg.eval.remasking.strategy,
            cfg.eval.remasking.eta,
            cfg.eval.remasking.get("t_switch", 1.0),
        )

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

    # Detokenize training set for comparison
    logger.info("Detokenizing training set for comparison metrics...")
    train_dicts = []
    for idx in range(len(train_ds)):
        item = train_ds[idx]
        try:
            gd = detokenize(item["tokens"], item["pad_mask"], vocab_config)
            train_dicts.append(gd)
        except ValueError:
            pass

    # --- Denoising eval (seed-independent, run once) ---
    denoising_metrics: dict[str, float] = {}
    run_denoising = cfg.eval.get("run_denoising_eval", False)
    if run_denoising:
        from torch.utils.data import DataLoader

        from bd_gen.eval.denoising_eval import denoising_eval, denoising_val_elbo

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
        val_loader = DataLoader(val_ds, batch_size=cfg.eval.batch_size, shuffle=False)

        t_grid = list(cfg.eval.get("denoising_t_grid", [0.1, 0.3, 0.5, 0.7, 0.9]))
        max_batches = cfg.eval.get("denoising_max_batches", None)
        logger.info("Running denoising eval (t_grid=%s)...", t_grid)

        denoising_metrics = denoising_eval(
            model, val_loader, noise_schedule, vocab_config,
            t_grid=t_grid, device=device, max_batches=max_batches,
        )
        for key, val in denoising_metrics.items():
            logger.info("  %s: %.4f", key, val)

    # --- Multi-seed evaluation ---
    seeds = list(cfg.eval.get("seeds", [cfg.seed]))
    logger.info("Running evaluation with %d seed(s): %s", len(seeds), seeds)

    per_seed_results: dict[int, dict] = {}
    for seed in seeds:
        logger.info("--- Seed %d ---", seed)
        seed_metrics = _generate_and_evaluate_single_seed(
            cfg=cfg,
            model=model,
            noise_schedule=noise_schedule,
            vocab_config=vocab_config,
            train_dicts=train_dicts,
            num_rooms_dist=num_rooms_dist,
            remasking_fn=remasking_fn,
            device=device,
            seed=seed,
        )
        per_seed_results[seed] = seed_metrics

        # Log per-seed scalar metrics
        scalar_keys = {
            k: v for k, v in seed_metrics.items() if isinstance(v, (int, float))
        }
        logger.info(
            "Seed %d — validity=%.1f%%, diversity=%.3f",
            seed,
            100 * scalar_keys.get("eval/validity_rate", 0),
            scalar_keys.get("eval/diversity", 0),
        )

    # --- Aggregate across seeds ---
    summary = _aggregate_multi_seed(per_seed_results)

    # Log summary means
    logger.info("--- Multi-seed summary (%d seeds) ---", len(seeds))
    for key in sorted(summary):
        s = summary[key]
        logger.info("  %s: %.4f +/- %.4f", key, s["mean"], s["std"])

    # --- Build flat metrics dict for wandb (summary means) ---
    flat_metrics: dict[str, float] = {}
    for key, s in summary.items():
        flat_metrics[key] = s["mean"]
        flat_metrics[f"{key}_std"] = s["std"]

    # Add denoising metrics (not aggregated, run once)
    flat_metrics.update(denoising_metrics)

    # Add prefixed aliases for wandb grouping
    prefixed = _prefix_metrics(flat_metrics)
    flat_metrics.update(prefixed)

    # --- Log to wandb ---
    log_metrics(flat_metrics)

    # --- Save results ---
    results_path = output_dir / "metrics.json"
    full_output = {
        "meta": {
            "checkpoint": str(ckpt_path.name),
            "num_samples": cfg.eval.num_samples,
            "sampling_steps": cfg.eval.sampling_steps,
            "seeds": seeds,
        },
        "per_seed": {
            str(s): _make_json_serializable(m)
            for s, m in per_seed_results.items()
        },
        "summary": summary,
        "denoising": denoising_metrics,
    }
    results_path.write_text(json.dumps(full_output, indent=2))
    logger.info("Metrics saved: %s", results_path)

    # --- Save structured result for cross-method comparison ---
    from eval_results.save_utils import save_eval_result  # noqa: E402

    remasking_cfg = cfg.eval.remasking
    unmasking_mode = cfg.eval.get("unmasking_mode", "random")
    top_p = cfg.eval.get("top_p", None)

    # Build systematic method name: {unmasking}_{sampling}_{remasking}
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

    eval_result_path = _PROJECT_ROOT / "eval_results" / f"{method_name}.json"
    save_eval_result(
        path=eval_result_path,
        method=method_name,
        config_dict={
            "seeds": seeds,
            "num_samples": cfg.eval.num_samples,
            "sampling_steps": cfg.eval.sampling_steps,
            "temperature": cfg.eval.temperature,
            "top_p": top_p,
            "unmasking_mode": unmasking_mode,
            "remasking_enabled": remasking_cfg.enabled,
            "remasking_strategy": (
                remasking_cfg.strategy if remasking_cfg.enabled
                else None
            ),
            "remasking_eta": (
                remasking_cfg.eta if remasking_cfg.enabled
                else None
            ),
            "remasking_t_switch": (
                remasking_cfg.get("t_switch", 1.0) if remasking_cfg.enabled
                else None
            ),
            "checkpoint": str(ckpt_path.name),
        },
        per_seed_metrics={
            s: _make_json_serializable(m)
            for s, m in per_seed_results.items()
        },
        summary_metrics=summary,
        denoising_metrics=denoising_metrics if denoising_metrics else None,
    )
    logger.info("Structured result saved: %s", eval_result_path)

    # --- Save samples (from last seed) ---
    if cfg.eval.save_samples:
        # Re-generate with last seed for reproducible sample saving
        last_seed = seeds[-1]
        set_seed(last_seed)
        viz_tokens = []
        with torch.no_grad():
            for start in range(0, min(cfg.eval.num_samples, cfg.eval.batch_size),
                               cfg.eval.batch_size):
                bs = min(cfg.eval.batch_size, cfg.eval.num_samples - start)
                batch_tokens = sample(
                    model=model,
                    noise_schedule=noise_schedule,
                    vocab_config=vocab_config,
                    batch_size=bs,
                    num_steps=cfg.eval.sampling_steps,
                    temperature=cfg.eval.temperature,
                    top_p=cfg.eval.get("top_p", None),
                    unmasking_mode=cfg.eval.get("unmasking_mode", "random"),
                    t_switch=cfg.eval.remasking.get("t_switch", 1.0),
                    remasking_fn=remasking_fn,
                    num_rooms_distribution=num_rooms_dist,
                    device=device,
                )
                viz_tokens.append(batch_tokens.cpu())
        viz_tokens = torch.cat(viz_tokens, dim=0)
        viz_pad = _reconstruct_pad_masks(viz_tokens, vocab_config)
        torch.save({"tokens": viz_tokens, "pad_masks": viz_pad},
                    output_dir / "samples.pt")

    # --- Visualize ---
    if cfg.eval.visualize:
        # Use samples from last seed
        last_seed_metrics = per_seed_results[seeds[-1]]
        # Re-detokenize for viz (use saved tokens if available)
        try:
            saved = torch.load(output_dir / "samples.pt", weights_only=True)
            viz_tokens_t = saved["tokens"]
            viz_pads_t = saved["pad_masks"]
            viz_dicts = []
            n_viz = min(cfg.eval.num_viz_samples, viz_tokens_t.size(0))
            for i in range(n_viz):
                try:
                    gd = detokenize(viz_tokens_t[i], viz_pads_t[i], vocab_config)
                    if gd["num_rooms"] > 0:
                        viz_dicts.append(gd)
                except ValueError:
                    pass
        except FileNotFoundError:
            viz_dicts = []

        if viz_dicts:
            fig = draw_bubble_diagram_grid(viz_dicts)
            fig_path = output_dir / "samples.png"
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            logger.info("Visualization saved: %s", fig_path)
            try:
                wandb.log({"eval/samples": wandb.Image(str(fig_path))})
            except Exception:
                pass
            import matplotlib.pyplot as plt
            plt.close(fig)

    wandb.finish()
    logger.info("Evaluation complete. Output: %s", output_dir)


def _make_json_serializable(obj):
    """Recursively convert numpy types and other non-serializable values."""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(v) for v in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


if __name__ == "__main__":
    cli_overrides = sys.argv[1:]
    config = _load_config(cli_overrides)
    evaluate(config)
