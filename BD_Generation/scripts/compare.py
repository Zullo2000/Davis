"""Generate comparison markdown from evaluation result JSONs.

Usage:
    python scripts/compare.py
    python scripts/compare.py --results eval_results/mdlm_baseline.json eval_results/remdm_cap_eta0.1.json
    python scripts/compare.py --output eval_results/comparison.md
    python scripts/compare.py --primary-only
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
        description="Generate comparison markdown from evaluation JSONs.",
    )
    parser.add_argument(
        "--results",
        nargs="+",
        type=str,
        default=None,
        help="Paths to result JSON files. Default: all *.json in eval_results/.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(_PROJECT_ROOT / "eval_results" / "comparison.md"),
        help="Output markdown path.",
    )
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="Only show primary metrics (JS/TV/W1), hide KL diagnostics.",
    )
    args = parser.parse_args()

    # Auto-discover JSON files if none specified
    if args.results is None:
        eval_dir = _PROJECT_ROOT / "eval_results"
        result_paths = sorted(eval_dir.glob("*.json"))
    else:
        result_paths = [Path(p) for p in args.results]

    if not result_paths:
        print("No result JSON files found.")
        sys.exit(1)

    print(f"Comparing {len(result_paths)} results:")
    for p in result_paths:
        print(f"  - {p.name}")

    md = build_comparison_table(result_paths, primary_only=args.primary_only)

    Path(args.output).write_text(md)
    print(f"\nComparison written to: {args.output}")


if __name__ == "__main__":
    main()
