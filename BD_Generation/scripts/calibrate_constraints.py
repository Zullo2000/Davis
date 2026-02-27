"""Calibrate constraint P90 normalizers from unguided samples (CPU only).

Loads saved sample tokens from ``eval_results/{schedule}/{model}_samples.pt``,
detokenizes them, computes hard violations for each constraint, and saves
the 90th-percentile normalizers to a JSON file.

Usage::

    python scripts/calibrate_constraints.py \
        --schedule loglinear_noise_sc \
        --model llada_topp0.9_no_remask \
        --constraints configs/guidance/example_basic.yaml \
        --output configs/guidance/calibration_basic.json
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

# Ensure BD_Generation is on sys.path when running as a script
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from bd_gen.data.tokenizer import detokenize  # noqa: E402
from bd_gen.data.vocab import VocabConfig  # noqa: E402
from bd_gen.guidance.calibration import (  # noqa: E402
    calibrate_from_samples,
    save_calibration,
)
from bd_gen.guidance.constraint_schema import (  # noqa: E402
    compile_constraints,
    load_guidance_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate constraint P90 normalizers from unguided samples.",
    )
    parser.add_argument(
        "--schedule", type=str, required=True,
        help="Noise schedule subdirectory (e.g., 'loglinear_noise_sc').",
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model name (e.g., 'llada_topp0.9_no_remask'). "
             "Loads {model}_samples.pt from eval_results/{schedule}/.",
    )
    parser.add_argument(
        "--constraints", type=str, required=True,
        help="Path to guidance constraint YAML/JSON config.",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output path for calibration JSON.",
    )
    parser.add_argument(
        "--n-max", type=int, default=8,
        help="Maximum rooms per graph (default: 8).",
    )
    args = parser.parse_args()

    # --- Locate samples file ---
    eval_dir = _PROJECT_ROOT / "eval_results" / args.schedule
    samples_path = eval_dir / f"{args.model}_samples.pt"
    if not samples_path.exists():
        print(f"Error: samples file not found: {samples_path}")
        sys.exit(1)

    # --- Load samples ---
    logger.info("Loading samples from %s", samples_path)
    data = torch.load(samples_path, weights_only=True)
    vocab_config = VocabConfig(n_max=data.get("n_max", args.n_max))
    seeds = data["seeds"]

    # --- Detokenize all samples across all seeds ---
    graph_dicts: list[dict] = []
    for seed in seeds:
        seed_key = str(seed)
        seed_data = data["per_seed"][seed_key]
        tokens = seed_data["tokens"]
        pad_masks = seed_data["pad_masks"]

        for i in range(tokens.size(0)):
            try:
                gd = detokenize(tokens[i], pad_masks[i], vocab_config)
                graph_dicts.append(gd)
            except ValueError:
                # Skip invalid samples
                pass

    logger.info("Detokenized %d samples across %d seeds", len(graph_dicts), len(seeds))

    # --- Load and compile constraints ---
    constraints_path = Path(args.constraints)
    if not constraints_path.is_absolute():
        constraints_path = _PROJECT_ROOT / constraints_path
    guidance_config = load_guidance_config(constraints_path)
    constraints = compile_constraints(guidance_config)
    logger.info("Loaded %d constraints from %s", len(constraints), constraints_path)

    # --- Calibrate ---
    calibration = calibrate_from_samples(graph_dicts, constraints)

    for name, p90 in calibration.items():
        logger.info("  %s: P90 = %.4f", name, p90)

    # --- Save ---
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = _PROJECT_ROOT / output_path
    save_calibration(output_path, calibration)
    logger.info("Calibration saved to %s", output_path)


if __name__ == "__main__":
    main()
