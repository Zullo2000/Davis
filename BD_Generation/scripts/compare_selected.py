"""Compare a selected subset of models on all metrics.

Usage:
    python scripts/compare_selected.py --schedule loglinear_noise_sc \
        --models llada_topp0.9_remdm_confidence_tsw0.5 llada_topp0.9_no_remask

    python scripts/compare_selected.py --schedule learned_noise_sc \
        --models v2_llada_topp0.9_no_remask \
        v2_llada_topp0.9_remdm_confidence_tsw1.0 \
        --primary-only

    # List available models for a schedule:
    python scripts/compare_selected.py --schedule loglinear_noise_sc --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from eval_results.save_utils import build_comparison_table  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a selected subset of models on all metrics.",
    )
    parser.add_argument(
        "--schedule",
        type=str,
        required=True,
        help="Noise schedule subdirectory (e.g., 'linear_noise_sc', 'loglinear_noise_sc', 'learned_noise_sc').",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=None,
        help="Model names to compare (without .json extension).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models for the given schedule and exit.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output markdown path. Default: prints to stdout.",
    )
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="Only show primary metrics (JS/TV/W1), hide KL diagnostics.",
    )
    parser.add_argument(
        "--guided",
        action="store_true",
        help="Use focused metric families for guidance experiments "
             "(Validity + Coverage + Priority Metrics).",
    )
    args = parser.parse_args()

    eval_dir = _PROJECT_ROOT / "eval_results" / args.schedule
    if not eval_dir.is_dir():
        print(f"Error: directory not found: {eval_dir}")
        sys.exit(1)

    available = sorted(p.stem for p in eval_dir.glob("*.json"))

    if args.list:
        print(f"Available models for schedule '{args.schedule}':")
        for name in available:
            print(f"  {name}")
        return

    if not args.models:
        print("Error: --models is required (or use --list to see available models).")
        sys.exit(1)

    # Resolve paths and validate
    result_paths: list[Path] = []
    for model in args.models:
        p = eval_dir / f"{model}.json"
        if not p.exists():
            print(f"Error: '{model}' not found in {eval_dir}")
            print(f"Available: {', '.join(available)}")
            sys.exit(1)
        result_paths.append(p)

    print(f"Comparing {len(result_paths)} models ({args.schedule} schedule):")
    for p in result_paths:
        print(f"  - {p.stem}")

    md = build_comparison_table(
        result_paths, primary_only=args.primary_only, guided=args.guided,
    )

    if args.output:
        Path(args.output).write_text(md)
        print(f"\nComparison written to: {args.output}")
    else:
        print()
        print(md)


if __name__ == "__main__":
    main()
