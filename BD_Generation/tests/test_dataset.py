"""Tests for bd_gen.data.dataset module.

Covers BubbleDiagramDataset construction, split logic, class weights,
num_rooms_distribution, __getitem__ correctness, and PAD invariant.

Uses synthetic mock data for fast unit tests and real data.mat for
integration tests when available.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from bd_gen.data.dataset import BubbleDiagramDataset
from bd_gen.data.vocab import (
    EDGE_NO_EDGE_IDX,
    EDGE_PAD_IDX,
    EDGE_VOCAB_SIZE,
    NODE_PAD_IDX,
    NODE_VOCAB_SIZE,
    RPLAN_VOCAB_CONFIG,
)

# ---------------------------------------------------------------------------
# Paths to real data (skip integration tests if absent)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MAT_PATH = _PROJECT_ROOT / "data" / "data.mat"
_CACHE_PATH = _PROJECT_ROOT / "data_cache" / "graph2plan_nmax8.pt"

needs_real_data = pytest.mark.skipif(
    not _MAT_PATH.exists(),
    reason=f"Real data.mat not found at {_MAT_PATH}",
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _make_synthetic_graphs(n: int = 100) -> list[dict]:
    """Build a list of synthetic graph dicts for testing.

    Produces deterministic graphs with 2-8 rooms each.
    """
    rng = np.random.RandomState(12345)
    graphs = []
    for _ in range(n):
        num_rooms = rng.randint(2, 9)  # 2..8
        node_types = rng.randint(0, 13, size=num_rooms).tolist()
        edge_triples = []
        for i in range(num_rooms):
            for j in range(i + 1, num_rooms):
                if rng.rand() < 0.4:
                    rel = rng.randint(0, 10)
                    edge_triples.append((i, j, rel))
        graphs.append(
            {
                "num_rooms": num_rooms,
                "node_types": node_types,
                "edge_triples": edge_triples,
            }
        )
    return graphs


def _make_dataset(
    tmp_path: Path,
    split: str = "train",
    n: int = 100,
    seed: int = 42,
) -> BubbleDiagramDataset:
    """Create a BubbleDiagramDataset backed by synthetic mock data."""
    graphs = _make_synthetic_graphs(n)
    mat_path = tmp_path / "data.mat"
    mat_path.touch()
    cache_path = tmp_path / "cache.pt"

    with patch(
        "bd_gen.data.dataset.load_graph2plan",
        return_value=graphs,
    ):
        ds = BubbleDiagramDataset(
            mat_path=mat_path,
            cache_path=cache_path,
            vocab_config=RPLAN_VOCAB_CONFIG,
            split=split,
            seed=seed,
        )
    return ds


# ===================================================================
# TestDatasetConstruction
# ===================================================================


class TestDatasetConstruction:
    """Test dataset creation and basic properties."""

    def test_train_split_size(self, tmp_path: Path):
        ds = _make_dataset(tmp_path, split="train", n=100)
        assert len(ds) == 80  # 0.8 * 100

    def test_val_split_size(self, tmp_path: Path):
        ds = _make_dataset(tmp_path, split="val", n=100)
        assert len(ds) == 10  # 0.1 * 100

    def test_test_split_size(self, tmp_path: Path):
        ds = _make_dataset(tmp_path, split="test", n=100)
        assert len(ds) == 10  # 0.1 * 100

    def test_splits_sum_to_total(self, tmp_path: Path):
        train = _make_dataset(tmp_path, split="train", n=100)
        val = _make_dataset(tmp_path, split="val", n=100)
        test = _make_dataset(tmp_path, split="test", n=100)
        assert len(train) + len(val) + len(test) == 100

    def test_invalid_split_raises(self, tmp_path: Path):
        with pytest.raises(ValueError, match="split"):
            _make_dataset(tmp_path, split="validation", n=10)

    def test_invalid_frac_sum_raises(self, tmp_path: Path):
        graphs = _make_synthetic_graphs(10)
        mat_path = tmp_path / "data.mat"
        mat_path.touch()
        cache_path = tmp_path / "cache.pt"

        with patch(
            "bd_gen.data.dataset.load_graph2plan",
            return_value=graphs,
        ):
            with pytest.raises(ValueError, match="sum to 1.0"):
                BubbleDiagramDataset(
                    mat_path=mat_path,
                    cache_path=cache_path,
                    vocab_config=RPLAN_VOCAB_CONFIG,
                    train_frac=0.5,
                    val_frac=0.1,
                    test_frac=0.1,
                )


# ===================================================================
# TestDeterminism
# ===================================================================


class TestDeterminism:
    """Splits are deterministic for the same seed."""

    def test_same_seed_same_split(self, tmp_path: Path):
        ds1 = _make_dataset(tmp_path, split="train", seed=99)
        ds2 = _make_dataset(tmp_path, split="train", seed=99)
        assert len(ds1) == len(ds2)
        for i in range(len(ds1)):
            assert torch.equal(ds1[i]["tokens"], ds2[i]["tokens"])

    def test_different_seed_different_split(self, tmp_path: Path):
        ds1 = _make_dataset(tmp_path, split="train", seed=1)
        ds2 = _make_dataset(tmp_path, split="train", seed=2)
        # At least one sample should differ
        any_differ = any(
            not torch.equal(ds1[i]["tokens"], ds2[i]["tokens"])
            for i in range(min(len(ds1), len(ds2)))
        )
        assert any_differ


# ===================================================================
# TestNoSplitOverlap
# ===================================================================


class TestNoSplitOverlap:
    """No sample appears in multiple splits."""

    def test_no_overlap(self, tmp_path: Path):
        train = _make_dataset(tmp_path, split="train", n=50, seed=42)
        val = _make_dataset(tmp_path, split="val", n=50, seed=42)
        test = _make_dataset(tmp_path, split="test", n=50, seed=42)

        # Collect token fingerprints (hash of token tensors)
        def fingerprints(ds):
            return {ds[i]["tokens"].numpy().tobytes() for i in range(len(ds))}

        train_fp = fingerprints(train)
        val_fp = fingerprints(val)
        test_fp = fingerprints(test)

        assert len(train_fp & val_fp) == 0, "train/val overlap"
        assert len(train_fp & test_fp) == 0, "train/test overlap"
        assert len(val_fp & test_fp) == 0, "val/test overlap"


# ===================================================================
# TestGetItem
# ===================================================================


class TestGetItem:
    """Verify __getitem__ return values."""

    def test_keys(self, tmp_path: Path):
        ds = _make_dataset(tmp_path, split="train", n=20)
        sample = ds[0]
        assert set(sample.keys()) == {"tokens", "pad_mask", "num_rooms"}

    def test_shapes(self, tmp_path: Path):
        ds = _make_dataset(tmp_path, split="train", n=20)
        vc = RPLAN_VOCAB_CONFIG
        sample = ds[0]
        assert sample["tokens"].shape == (vc.seq_len,)
        assert sample["pad_mask"].shape == (vc.seq_len,)

    def test_dtypes(self, tmp_path: Path):
        ds = _make_dataset(tmp_path, split="train", n=20)
        sample = ds[0]
        assert sample["tokens"].dtype == torch.long
        assert sample["pad_mask"].dtype == torch.bool
        assert isinstance(sample["num_rooms"], int)

    def test_num_rooms_range(self, tmp_path: Path):
        ds = _make_dataset(tmp_path, split="train", n=100)
        for i in range(len(ds)):
            nr = ds[i]["num_rooms"]
            assert 1 <= nr <= RPLAN_VOCAB_CONFIG.n_max

    def test_consistent_access(self, tmp_path: Path):
        """Same index returns same data on repeated access."""
        ds = _make_dataset(tmp_path, split="train", n=20)
        s1 = ds[0]
        s2 = ds[0]
        assert torch.equal(s1["tokens"], s2["tokens"])
        assert torch.equal(s1["pad_mask"], s2["pad_mask"])
        assert s1["num_rooms"] == s2["num_rooms"]


# ===================================================================
# TestPadInvariant
# ===================================================================


class TestPadInvariant:
    """PAD positions have PAD tokens, real positions never do."""

    def test_pad_positions_have_pad_tokens(self, tmp_path: Path):
        ds = _make_dataset(tmp_path, split="train", n=50)
        vc = RPLAN_VOCAB_CONFIG
        for i in range(len(ds)):
            sample = ds[i]
            tokens = sample["tokens"]
            mask = sample["pad_mask"]
            for idx in range(vc.seq_len):
                if not mask[idx]:
                    val = int(tokens[idx].item())
                    if idx < vc.n_max:
                        assert val == NODE_PAD_IDX, (
                            f"Sample {i}, pos {idx}: "
                            f"expected NODE_PAD_IDX, got {val}"
                        )
                    else:
                        assert val == EDGE_PAD_IDX, (
                            f"Sample {i}, pos {idx}: "
                            f"expected EDGE_PAD_IDX, got {val}"
                        )

    def test_real_positions_never_pad(self, tmp_path: Path):
        ds = _make_dataset(tmp_path, split="train", n=50)
        vc = RPLAN_VOCAB_CONFIG
        for i in range(len(ds)):
            sample = ds[i]
            tokens = sample["tokens"]
            mask = sample["pad_mask"]
            for idx in range(vc.seq_len):
                if mask[idx]:
                    val = int(tokens[idx].item())
                    if idx < vc.n_max:
                        assert val != NODE_PAD_IDX
                    else:
                        assert val != EDGE_PAD_IDX


# ===================================================================
# TestClassWeights
# ===================================================================


class TestClassWeights:
    """Verify class weight computation."""

    def test_edge_weights_shape_and_dtype(self, tmp_path: Path):
        ds = _make_dataset(tmp_path, split="train", n=100)
        w = ds.edge_class_weights
        assert w is not None
        assert w.shape == (EDGE_VOCAB_SIZE,)
        assert w.dtype == torch.float32

    def test_node_weights_shape_and_dtype(self, tmp_path: Path):
        ds = _make_dataset(tmp_path, split="train", n=100)
        w = ds.node_class_weights
        assert w is not None
        assert w.shape == (NODE_VOCAB_SIZE,)
        assert w.dtype == torch.float32

    def test_edge_weights_all_nonnegative(self, tmp_path: Path):
        ds = _make_dataset(tmp_path, split="train", n=100)
        assert (ds.edge_class_weights >= 0).all()

    def test_node_weights_all_nonnegative(self, tmp_path: Path):
        ds = _make_dataset(tmp_path, split="train", n=100)
        assert (ds.node_class_weights >= 0).all()

    def test_edge_pad_class_weight_zero(self, tmp_path: Path):
        """PAD token class should have zero weight (never appears in real)."""
        ds = _make_dataset(tmp_path, split="train", n=100)
        assert ds.edge_class_weights[EDGE_PAD_IDX] == 0.0

    def test_node_pad_class_weight_zero(self, tmp_path: Path):
        ds = _make_dataset(tmp_path, split="train", n=100)
        assert ds.node_class_weights[NODE_PAD_IDX] == 0.0

    def test_val_test_have_no_weights(self, tmp_path: Path):
        val = _make_dataset(tmp_path, split="val", n=50)
        test = _make_dataset(tmp_path, split="test", n=50)
        assert val.edge_class_weights is None
        assert val.node_class_weights is None
        assert test.edge_class_weights is None
        assert test.node_class_weights is None

    def test_real_edge_types_have_positive_weight(self, tmp_path: Path):
        """Edge types 0-9 and NO_EDGE should have positive weights."""
        ds = _make_dataset(tmp_path, split="train", n=200)
        for c in range(EDGE_NO_EDGE_IDX + 1):
            assert ds.edge_class_weights[c] > 0, (
                f"Edge class {c} has zero weight"
            )


# ===================================================================
# TestNumRoomsDistribution
# ===================================================================


class TestNumRoomsDistribution:
    """Verify num_rooms_distribution computation."""

    def test_shape(self, tmp_path: Path):
        ds = _make_dataset(tmp_path, split="train", n=100)
        dist = ds.num_rooms_distribution
        assert dist is not None
        assert dist.shape == (RPLAN_VOCAB_CONFIG.n_max,)

    def test_dtype(self, tmp_path: Path):
        ds = _make_dataset(tmp_path, split="train", n=100)
        assert ds.num_rooms_distribution.dtype == torch.float32

    def test_sums_to_one(self, tmp_path: Path):
        ds = _make_dataset(tmp_path, split="train", n=100)
        assert abs(ds.num_rooms_distribution.sum().item() - 1.0) < 1e-5

    def test_all_nonnegative(self, tmp_path: Path):
        ds = _make_dataset(tmp_path, split="train", n=100)
        assert (ds.num_rooms_distribution >= 0).all()

    def test_val_test_have_no_distribution(self, tmp_path: Path):
        val = _make_dataset(tmp_path, split="val", n=50)
        test = _make_dataset(tmp_path, split="test", n=50)
        assert val.num_rooms_distribution is None
        assert test.num_rooms_distribution is None


# ===================================================================
# TestWithRealData (integration)
# ===================================================================


@needs_real_data
class TestWithRealData:
    """Integration tests with the real Graph2Plan dataset."""

    @pytest.fixture(scope="class")
    def train_ds(self) -> BubbleDiagramDataset:
        return BubbleDiagramDataset(
            mat_path=_MAT_PATH,
            cache_path=_CACHE_PATH,
            vocab_config=RPLAN_VOCAB_CONFIG,
            split="train",
        )

    def test_size_reasonable(self, train_ds: BubbleDiagramDataset):
        """Expect ~64K training samples (80% of ~80K)."""
        assert len(train_ds) > 50_000

    def test_getitem_works(self, train_ds: BubbleDiagramDataset):
        sample = train_ds[0]
        assert sample["tokens"].shape == (RPLAN_VOCAB_CONFIG.seq_len,)

    def test_edge_class_weights_present(self, train_ds: BubbleDiagramDataset):
        assert train_ds.edge_class_weights is not None
        assert train_ds.edge_class_weights.shape == (EDGE_VOCAB_SIZE,)

    def test_num_rooms_distribution_present(
        self, train_ds: BubbleDiagramDataset
    ):
        assert train_ds.num_rooms_distribution is not None
        assert abs(train_ds.num_rooms_distribution.sum().item() - 1.0) < 1e-5

    def test_dataloader_iteration(self, train_ds: BubbleDiagramDataset):
        """Smoke test: DataLoader can iterate a batch."""
        loader = torch.utils.data.DataLoader(
            train_ds, batch_size=4, shuffle=False
        )
        batch = next(iter(loader))
        assert batch["tokens"].shape == (4, RPLAN_VOCAB_CONFIG.seq_len)
        assert batch["pad_mask"].shape == (4, RPLAN_VOCAB_CONFIG.seq_len)
        assert len(batch["num_rooms"]) == 4
