"""One-time data preparation script for Graph2Plan dataset.

Downloads Data.zip from the Graph2Plan GitHub release, extracts
data.mat, parses it into the cached graph list, and prints summary
statistics.

Usage:
    python -m scripts.prepare_data

    Or from the BD_Generation directory:
    python scripts/prepare_data.py

This script is NOT Hydra-decorated. It reads paths from
configs/data/graph2plan.yaml via OmegaConf.load().
"""

from __future__ import annotations

import logging
import sys
import zipfile
from collections import Counter
from pathlib import Path
from urllib.request import urlretrieve

from omegaconf import OmegaConf
from tqdm import tqdm

# Ensure BD_Generation is on sys.path when running as a script
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from bd_gen.data.graph2plan_loader import load_graph2plan  # noqa: E402
from bd_gen.data.vocab import EDGE_TYPES, NODE_TYPES  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------


class _TqdmUpTo(tqdm):
    """tqdm wrapper for urlretrieve reporthook."""

    def update_to(
        self,
        blocks: int = 1,
        block_size: int = 1,
        total_size: int = -1,
    ) -> None:
        """Update progress bar.

        Args:
            blocks: Number of blocks transferred so far.
            block_size: Size of each block in bytes.
            total_size: Total size in bytes (-1 if unknown).
        """
        if total_size > 0:
            self.total = total_size
        self.update(blocks * block_size - self.n)


def _download_and_extract(
    url: str,
    mat_path: Path,
) -> None:
    """Download Data.zip and extract data.mat from it.

    Args:
        url: URL to the Data.zip release asset.
        mat_path: Destination path for the extracted data.mat.
    """
    mat_path.parent.mkdir(parents=True, exist_ok=True)
    zip_path = mat_path.parent / "Data.zip"

    # Download
    logger.info("Downloading %s ...", url)
    with _TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc="Data.zip",
    ) as pbar:
        urlretrieve(url, str(zip_path), reporthook=pbar.update_to)

    logger.info("Download complete: %s", zip_path)

    # Extract data.mat
    logger.info("Extracting data.mat from %s ...", zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Find data.mat inside the zip (may be nested in a subdirectory)
        mat_names = [n for n in zf.namelist() if n.endswith("data.mat")]
        if not mat_names:
            raise FileNotFoundError(
                f"data.mat not found inside {zip_path}. "
                f"Contents: {zf.namelist()[:20]}"
            )
        mat_name = mat_names[0]
        logger.info("Found %s in archive", mat_name)

        # Extract to the target location
        with zf.open(mat_name) as src, open(mat_path, "wb") as dst:
            dst.write(src.read())

    logger.info("Extracted data.mat to %s", mat_path)

    # Clean up zip
    zip_path.unlink()
    logger.info("Removed %s", zip_path)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def _print_summary(graphs: list[dict]) -> None:
    """Print summary statistics for the parsed dataset.

    Args:
        graphs: List of graph dicts from load_graph2plan.
    """
    print("\n" + "=" * 60)
    print("Graph2Plan Dataset Summary")
    print("=" * 60)
    print(f"Total graphs: {len(graphs):,}")

    # Room count distribution
    room_counts = Counter(g["num_rooms"] for g in graphs)
    print("\nRoom count distribution:")
    for n in sorted(room_counts):
        pct = 100.0 * room_counts[n] / len(graphs)
        bar = "#" * int(pct / 2)
        print(f"  {n} rooms: {room_counts[n]:>6,}  ({pct:5.1f}%)  {bar}")

    # Edge count stats
    edge_counts = [len(g["edge_triples"]) for g in graphs]
    print(
        f"\nEdges per graph: "
        f"min={min(edge_counts)}, "
        f"max={max(edge_counts)}, "
        f"mean={sum(edge_counts) / len(edge_counts):.1f}, "
        f"total={sum(edge_counts):,}"
    )

    # Edge type distribution
    edge_type_counts = Counter()
    for g in graphs:
        for _u, _v, rel in g["edge_triples"]:
            edge_type_counts[rel] += 1

    print("\nEdge type distribution:")
    total_edges = sum(edge_type_counts.values())
    for idx in range(len(EDGE_TYPES)):
        count = edge_type_counts.get(idx, 0)
        pct = 100.0 * count / total_edges if total_edges > 0 else 0
        print(f"  {idx:2d} ({EDGE_TYPES[idx]:<14s}): {count:>8,}  ({pct:5.1f}%)")

    # Node type distribution
    node_type_counts = Counter()
    for g in graphs:
        for nt in g["node_types"]:
            node_type_counts[nt] += 1

    print("\nNode type distribution:")
    total_nodes = sum(node_type_counts.values())
    for idx in range(len(NODE_TYPES)):
        count = node_type_counts.get(idx, 0)
        pct = 100.0 * count / total_nodes if total_nodes > 0 else 0
        print(f"  {idx:2d} ({NODE_TYPES[idx]:<14s}): {count:>8,}  ({pct:5.1f}%)")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Download (if needed), parse, cache, and summarize Graph2Plan data."""
    # Load config
    config_path = _PROJECT_ROOT / "configs" / "data" / "graph2plan.yaml"
    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        sys.exit(1)

    cfg = OmegaConf.load(config_path)

    mat_path = _PROJECT_ROOT / cfg.mat_path
    cache_path = _PROJECT_ROOT / cfg.cache_path
    n_max = cfg.n_max
    mat_url = cfg.mat_url

    logger.info("Configuration:")
    logger.info("  mat_path:   %s", mat_path)
    logger.info("  cache_path: %s", cache_path)
    logger.info("  n_max:      %d", n_max)

    # Download if needed
    if not mat_path.exists():
        logger.info("data.mat not found, downloading...")
        _download_and_extract(mat_url, mat_path)
    else:
        logger.info("data.mat already exists at %s", mat_path)

    # Parse and cache
    graphs = load_graph2plan(mat_path, cache_path, n_max=n_max)

    # Print summary
    _print_summary(graphs)


if __name__ == "__main__":
    main()
