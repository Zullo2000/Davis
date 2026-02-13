"""BubbleDiagramDataset: PyTorch dataset for tokenized bubble diagrams.

Loads graphs via :func:`graph2plan_loader.load_graph2plan`, tokenizes all
samples upfront, creates deterministic train/val/test splits, and computes
class weights and room-count distribution from the training split.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from bd_gen.data.graph2plan_loader import load_graph2plan
from bd_gen.data.tokenizer import tokenize
from bd_gen.data.vocab import (
    EDGE_PAD_IDX,
    EDGE_VOCAB_SIZE,
    NODE_PAD_IDX,
    NODE_VOCAB_SIZE,
    VocabConfig,
)


class BubbleDiagramDataset(Dataset):
    """Dataset of tokenized bubble diagram graphs.

    All graphs are tokenized once during ``__init__`` and stored as
    tensors in RAM (~26 MB for 80K RPLAN samples). Per-sample access
    is O(1).

    Attributes:
        vocab_config: The VocabConfig controlling sequence layout.
        split: Which split this instance represents.
        edge_class_weights: Inverse-frequency weights for edge classes,
            shape ``(EDGE_VOCAB_SIZE,)``. Computed from the training
            split only; ``None`` for val/test splits.
        node_class_weights: Inverse-frequency weights for node classes,
            shape ``(NODE_VOCAB_SIZE,)``. Computed from the training
            split only; ``None`` for val/test splits.
        num_rooms_distribution: Normalised histogram of room counts in
            the training split, shape ``(n_max,)``. Index ``k``
            corresponds to ``k + 1`` rooms. ``None`` for val/test.
    """

    def __init__(
        self,
        mat_path: str | Path,
        cache_path: str | Path,
        vocab_config: VocabConfig,
        split: str = "train",
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        seed: int = 42,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(
                f"split must be 'train', 'val', or 'test', got '{split}'"
            )
        total_frac = train_frac + val_frac + test_frac
        if abs(total_frac - 1.0) > 1e-6:
            raise ValueError(
                f"Split fractions must sum to 1.0, got {total_frac:.6f}"
            )

        self.vocab_config = vocab_config
        self.split = split

        # --- Load and tokenize all graphs ---
        graphs = load_graph2plan(mat_path, cache_path, n_max=vocab_config.n_max)
        n_total = len(graphs)

        all_tokens = torch.zeros(n_total, vocab_config.seq_len, dtype=torch.long)
        all_masks = torch.zeros(n_total, vocab_config.seq_len, dtype=torch.bool)
        all_num_rooms = torch.zeros(n_total, dtype=torch.long)

        for i, g in enumerate(graphs):
            tokens, pad_mask = tokenize(g, vocab_config)
            all_tokens[i] = tokens
            all_masks[i] = pad_mask
            all_num_rooms[i] = g["num_rooms"]

        # --- Deterministic split ---
        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n_total, generator=generator)

        n_train = int(train_frac * n_total)
        n_val = int(val_frac * n_total)

        if split == "train":
            indices = perm[:n_train]
        elif split == "val":
            indices = perm[n_train : n_train + n_val]
        else:  # test
            indices = perm[n_train + n_val :]

        self._tokens = all_tokens[indices]
        self._pad_masks = all_masks[indices]
        self._num_rooms = all_num_rooms[indices]

        # --- Class weights & distribution (training split only) ---
        if split == "train":
            self.edge_class_weights = self._compute_class_weights(
                self._tokens,
                self._pad_masks,
                start=vocab_config.n_max,
                end=vocab_config.seq_len,
                vocab_size=EDGE_VOCAB_SIZE,
                pad_idx=EDGE_PAD_IDX,
            )
            self.node_class_weights = self._compute_class_weights(
                self._tokens,
                self._pad_masks,
                start=0,
                end=vocab_config.n_max,
                vocab_size=NODE_VOCAB_SIZE,
                pad_idx=NODE_PAD_IDX,
            )
            self.num_rooms_distribution = self._compute_num_rooms_distribution(
                self._num_rooms, vocab_config.n_max
            )
        else:
            self.edge_class_weights = None
            self.node_class_weights = None
            self.num_rooms_distribution = None

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._tokens.shape[0]

    def __getitem__(self, idx: int) -> dict:
        """Return a single sample.

        Returns:
            Dict with keys ``"tokens"`` (long), ``"pad_mask"`` (bool),
            and ``"num_rooms"`` (int).
        """
        return {
            "tokens": self._tokens[idx],
            "pad_mask": self._pad_masks[idx],
            "num_rooms": int(self._num_rooms[idx].item()),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_class_weights(
        tokens: Tensor,
        pad_masks: Tensor,
        start: int,
        end: int,
        vocab_size: int,
        pad_idx: int,
    ) -> Tensor:
        """Compute inverse-frequency class weights over non-PAD positions.

        Args:
            tokens: ``(N, seq_len)`` token tensor.
            pad_masks: ``(N, seq_len)`` boolean mask (True = real).
            start: First column index of the region to count.
            end: One-past-last column index.
            vocab_size: Total number of classes.
            pad_idx: Token index representing PAD (excluded).

        Returns:
            Tensor of shape ``(vocab_size,)`` with inverse-frequency
            weights. Classes that never appear get weight 0.
        """
        region_tokens = tokens[:, start:end]
        region_masks = pad_masks[:, start:end]

        # Only count non-PAD positions
        real_tokens = region_tokens[region_masks]
        counts = torch.zeros(vocab_size, dtype=torch.long)
        for c in range(vocab_size):
            counts[c] = (real_tokens == c).sum()

        total = counts.sum()
        weights = torch.zeros(vocab_size, dtype=torch.float32)
        for c in range(vocab_size):
            if counts[c] > 0:
                weights[c] = total.float() / (vocab_size * counts[c].float())

        return weights

    @staticmethod
    def _compute_num_rooms_distribution(
        num_rooms: Tensor,
        n_max: int,
    ) -> Tensor:
        """Compute normalised room-count histogram.

        Args:
            num_rooms: ``(N,)`` tensor of room counts (1-based).
            n_max: Maximum rooms (histogram has n_max bins).

        Returns:
            Tensor of shape ``(n_max,)`` summing to 1.0.
            Index ``k`` = probability of ``k + 1`` rooms.
        """
        dist = torch.zeros(n_max, dtype=torch.float32)
        for k in range(n_max):
            dist[k] = (num_rooms == (k + 1)).sum().float()
        dist = dist / dist.sum()
        return dist
