"""Compute metrics from saved sample tokens (CPU only, no model needed).

Loads ``{method}_samples.pt`` files produced by ``generate_samples.py``,
detokenizes, computes all metrics, and saves/updates ``{method}.json``.

Usage::

    # Evaluate one model
    python scripts/evaluate.py --schedule loglinear --model llada_argmax_no_remask

    # Evaluate all models with saved samples
    python scripts/evaluate.py --schedule loglinear

    # List models with/without saved samples
    python scripts/evaluate.py --schedule loglinear --list

    # Also regenerate comparison.md
    python scripts/evaluate.py --schedule loglinear --update-comparison
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

# Ensure BD_Generation is on sys.path when running as a script
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from bd_gen.data.dataset import BubbleDiagramDataset  # noqa: E402
from bd_gen.data.tokenizer import detokenize  # noqa: E402
from bd_gen.data.vocab import VocabConfig  # noqa: E402
from bd_gen.eval.metrics import (  # noqa: E402
    conditional_edge_distances_topN,
    conditional_edge_kl,
    distribution_match,
    diversity,
    edge_present_rate_by_num_rooms,
    graph_structure_mmd,
    inside_validity,
    mode_coverage,
    novelty,
    spatial_transitivity,
    spatial_transitivity_by_num_rooms,
    type_conditioned_degree_kl,
    validity_by_num_rooms,
    validity_rate,
)
from bd_gen.eval.validity import check_validity_batch  # noqa: E402
from eval_results.save_utils import (  # noqa: E402
    aggregate_multi_seed,
    build_comparison_table,
    make_json_serializable,
    save_eval_result,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------


def compute_all_metrics(
    tokens: torch.Tensor,
    pad_masks: torch.Tensor,
    vocab_config: VocabConfig,
    train_dicts: list[dict],
    n_max: int,
    conditional_topN_pairs: int | None = 20,
) -> dict:
    """Compute all metrics from tokens + pad_masks.

    This is a pure-CPU function.  It detokenizes, runs validity checks,
    and computes every metric unconditionally.

    Returns a flat dict of metric_name -> float (or nested dicts for
    stratified metrics).
    """
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

    # --- All metrics (unconditional — always compute everything) ---
    metrics: dict = {"eval/validity_rate": v_rate}

    metrics["eval/diversity"] = diversity(graph_dicts)
    metrics["eval/novelty"] = novelty(graph_dicts, train_dicts)

    dm = distribution_match(graph_dicts, train_dicts)
    for key, val in dm.items():
        metrics[f"eval/{key}"] = val

    cekl = conditional_edge_kl(graph_dicts, train_dicts)
    for key, val in cekl.items():
        metrics[f"eval/{key}"] = val

    if conditional_topN_pairs is not None:
        topn_result = conditional_edge_distances_topN(
            graph_dicts, train_dicts, top_n=conditional_topN_pairs,
        )
        for key, val in topn_result.items():
            metrics[f"eval/{key}"] = val

    mmd = graph_structure_mmd(graph_dicts, train_dicts, n_max=n_max)
    for key, val in mmd.items():
        metrics[f"eval/{key}"] = val

    st = spatial_transitivity(graph_dicts)
    for key, val in st.items():
        metrics[f"eval/{key}"] = val

    tcdkl = type_conditioned_degree_kl(graph_dicts, train_dicts, n_max=n_max)
    for key, val in tcdkl.items():
        metrics[f"eval/{key}"] = val

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

    metrics["eval/inside_validity"] = inside_validity(graph_dicts)

    # --- Stratified metrics ---
    metrics["validity_by_num_rooms"] = validity_by_num_rooms(
        validity_results, graph_dicts,
    )
    metrics["transitivity_by_num_rooms"] = spatial_transitivity_by_num_rooms(
        graph_dicts,
    )
    metrics["edge_present_rate_by_num_rooms"] = edge_present_rate_by_num_rooms(
        graph_dicts,
    )

    return metrics


# ---------------------------------------------------------------------------
# Single-method evaluation
# ---------------------------------------------------------------------------


def evaluate_method(
    samples_path: Path,
    eval_dir: Path,
    train_dicts: list[dict],
    n_max: int = 8,
    conditional_topN_pairs: int | None = 20,
) -> dict[str, dict[str, float]]:
    """Evaluate a single method from saved sample tensors.

    Loads the ``_samples.pt``, runs all metrics per seed, aggregates,
    and writes/updates the corresponding ``.json`` result file.

    Returns the summary dict.
    """
    data = torch.load(samples_path, weights_only=True)
    vocab_config = VocabConfig(n_max=data.get("n_max", n_max))
    seeds = data["seeds"]
    method = data.get("method", samples_path.stem.replace("_samples", ""))

    # Preserve config and denoising from existing JSON if present
    json_path = eval_dir / f"{method}.json"
    config_dict = data.get("config", {})
    denoising_metrics: dict = {}
    if json_path.exists():
        existing = json.loads(json_path.read_text())
        config_dict = existing.get("config", config_dict)
        denoising_metrics = existing.get("denoising", {})

    # Evaluate each seed
    per_seed_results: dict[int, dict] = {}
    for seed in seeds:
        seed_key = str(seed)
        seed_data = data["per_seed"][seed_key]
        tokens = seed_data["tokens"]
        pad_masks = seed_data["pad_masks"]

        logger.info(
            "  Seed %s: %d samples, seq_len=%d",
            seed_key, tokens.size(0), tokens.size(1),
        )

        metrics = compute_all_metrics(
            tokens, pad_masks, vocab_config, train_dicts,
            n_max, conditional_topN_pairs,
        )
        per_seed_results[int(seed)] = metrics

    # Aggregate across seeds
    summary = aggregate_multi_seed(per_seed_results)

    # Log key metrics
    vr = summary.get("eval/validity_rate", {})
    div = summary.get("eval/diversity", {})
    logger.info(
        "  validity=%.1f%% +/- %.1f%%, diversity=%.3f +/- %.3f",
        100 * vr.get("mean", 0), 100 * vr.get("std", 0),
        div.get("mean", 0), div.get("std", 0),
    )

    # Save result JSON
    save_eval_result(
        path=json_path,
        method=method,
        config_dict=config_dict,
        per_seed_metrics={
            s: make_json_serializable(m) for s, m in per_seed_results.items()
        },
        summary_metrics=summary,
        denoising_metrics=denoising_metrics if denoising_metrics else None,
    )
    logger.info("  Saved: %s", json_path)
    return summary


# ---------------------------------------------------------------------------
# Training data loading
# ---------------------------------------------------------------------------


def load_train_dicts(n_max: int = 8) -> list[dict]:
    """Load and detokenize the training set for reference distributions."""
    vocab_config = VocabConfig(n_max=n_max)
    mat_path = _PROJECT_ROOT / "data" / "data.mat"
    cache_path = _PROJECT_ROOT / "data_cache" / f"graph2plan_nmax{n_max}.pt"

    train_ds = BubbleDiagramDataset(
        mat_path=mat_path,
        cache_path=cache_path,
        vocab_config=vocab_config,
        split="train",
        train_frac=0.8,
        val_frac=0.1,
        test_frac=0.1,
        seed=42,
    )

    logger.info("Detokenizing training set (%d samples)...", len(train_ds))
    train_dicts = []
    for idx in range(len(train_ds)):
        item = train_ds[idx]
        try:
            gd = detokenize(item["tokens"], item["pad_mask"], vocab_config)
            train_dicts.append(gd)
        except ValueError:
            pass
    logger.info("Training set: %d valid graphs", len(train_dicts))
    return train_dicts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute metrics from saved sample tokens (CPU only).",
    )
    parser.add_argument(
        "--schedule", type=str, required=True,
        help="Noise schedule subdirectory (e.g., 'loglinear', 'linear').",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Single model name to evaluate (without extension).",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List models with/without saved samples and exit.",
    )
    parser.add_argument(
        "--n-max", type=int, default=8,
        help="Maximum rooms per graph (default: 8).",
    )
    parser.add_argument(
        "--update-comparison", action="store_true",
        help="Regenerate comparison.md after evaluation.",
    )
    parser.add_argument(
        "--conditional-topn", type=int, default=20,
        help="Top-N pairs for conditional metrics (default: 20).",
    )
    args = parser.parse_args()

    eval_dir = _PROJECT_ROOT / "eval_results" / args.schedule
    if not eval_dir.is_dir():
        print(f"Error: directory not found: {eval_dir}")
        sys.exit(1)

    # Find models with saved samples
    sample_files = sorted(eval_dir.glob("*_samples.pt"))
    available = {p.stem.replace("_samples", ""): p for p in sample_files}

    if args.list:
        print(f"\nModels WITH saved samples ({args.schedule}):")
        for name in sorted(available):
            size_mb = available[name].stat().st_size / 1e6
            print(f"  {name} ({size_mb:.1f} MB)")

        json_files = sorted(eval_dir.glob("*.json"))
        no_samples = [
            p.stem for p in json_files
            if p.stem not in available and p.stem != "comparison"
        ]
        if no_samples:
            print("\nModels WITHOUT saved samples (need generate_samples.py):")
            for name in no_samples:
                print(f"  {name}")
        if not available and not no_samples:
            print("  (none)")
        return

    # Determine targets
    if args.model:
        if args.model not in available:
            print(f"Error: no samples found for '{args.model}'")
            if available:
                print(f"Available: {', '.join(sorted(available))}")
            sys.exit(1)
        targets = {args.model: available[args.model]}
    else:
        targets = available

    if not targets:
        print(f"No sample files found in {eval_dir}")
        sys.exit(1)

    # Load training data (one-time)
    train_dicts = load_train_dicts(n_max=args.n_max)

    # Evaluate each target
    for method_name, samples_pt in sorted(targets.items()):
        logger.info("Evaluating: %s", method_name)
        evaluate_method(
            samples_path=samples_pt,
            eval_dir=eval_dir,
            train_dicts=train_dicts,
            n_max=args.n_max,
            conditional_topN_pairs=args.conditional_topn,
        )

    # Optionally regenerate comparison table
    if args.update_comparison:
        json_files = sorted(eval_dir.glob("*.json"))
        result_jsons = [p for p in json_files if p.stem != "comparison"]
        if result_jsons:
            md = build_comparison_table(result_jsons)
            comp_path = eval_dir / "comparison.md"
            comp_path.write_text(md)
            logger.info("Comparison updated: %s", comp_path)


if __name__ == "__main__":
    main()
