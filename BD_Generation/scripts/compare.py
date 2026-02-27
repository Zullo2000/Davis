"""Generate comparison markdown from evaluation result JSONs.

Usage:
    python scripts/compare.py --schedule linear_noise_sc
    python scripts/compare.py --schedule loglinear_noise_sc
    python scripts/compare.py --results eval_results/linear_noise_sc/foo.json eval_results/linear_noise_sc/bar.json
    python scripts/compare.py --schedule linear_noise_sc --primary-only
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
        "--schedule",
        type=str,
        default=None,
        help="Noise schedule subdirectory (e.g., 'linear_noise_sc', 'loglinear_noise_sc', 'learned_noise_sc'). "
        "Auto-discovers *.json in eval_results/<schedule>/.",
    )
    parser.add_argument(
        "--results",
        nargs="+",
        type=str,
        default=None,
        help="Explicit paths to result JSON files (overrides --schedule).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output markdown path. Default: eval_results/<schedule>/comparison.md.",
    )
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help="Only show primary metrics (JS/TV/W1), hide KL diagnostics.",
    )
    args = parser.parse_args()

    # Determine result paths
    if args.results is not None:
        result_paths = [Path(p) for p in args.results]
        output_path = args.output or str(
            _PROJECT_ROOT / "eval_results" / "comparison.md"
        )
    elif args.schedule is not None:
        eval_dir = _PROJECT_ROOT / "eval_results" / args.schedule
        result_paths = sorted(eval_dir.glob("*.json"))
        output_path = args.output or str(eval_dir / "comparison.md")
    else:
        # Fallback: try all schedule subdirectories
        eval_root = _PROJECT_ROOT / "eval_results"
        result_paths = []
        for subdir in sorted(eval_root.iterdir()):
            if subdir.is_dir() and subdir.name != "__pycache__":
                sub_results = sorted(subdir.glob("*.json"))
                if sub_results:
                    print(f"\n=== Schedule: {subdir.name} ===")
                    md = build_comparison_table(
                        sub_results, primary_only=args.primary_only,
                    )
                    out = args.output or str(subdir / "comparison.md")
                    Path(out).write_text(md)
                    print(f"Comparison written to: {out}")
        if not result_paths:
            return
        output_path = args.output or str(
            _PROJECT_ROOT / "eval_results" / "comparison.md"
        )

    if not result_paths:
        print("No result JSON files found.")
        sys.exit(1)

    print(f"Comparing {len(result_paths)} results:")
    for p in result_paths:
        print(f"  - {p.name}")

    md = build_comparison_table(result_paths, primary_only=args.primary_only)

    Path(output_path).write_text(md)
    print(f"\nComparison written to: {output_path}")


if __name__ == "__main__":
    main()
