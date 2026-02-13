"""Vocabulary definitions for bubble diagram token sequences.

This module is the single source of truth for all vocabulary constants,
index mappings, and dataset-dependent sizing. All other modules import
from here rather than defining their own constants.

VERIFIED (Phase 1): NODE_TYPES and EDGE_TYPES match Graph2Plan's
get_vocab() in Network/model/utils.py. Data uses 0-based indexing.
Graph2Plan defines 15 room types (0-14) but only 0-12 appear in the
bubble diagram data — indices 13 (External) and 14 (ExteriorWall) are
unused, so we repurpose them as MASK and PAD tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Node vocabulary (N_MAX-independent)
# ---------------------------------------------------------------------------

NODE_TYPES: Final[list[str]] = [
    "LivingRoom",       # 0  — every floorplan has exactly one
    "MasterRoom",       # 1
    "Kitchen",          # 2
    "Bathroom",         # 3
    "DiningRoom",       # 4
    "ChildRoom",        # 5
    "StudyRoom",        # 6
    "SecondRoom",       # 7  — most common type (multiple bedrooms)
    "GuestRoom",        # 8
    "Balcony",          # 9
    "Entrance",         # 10
    "Storage",          # 11
    "Wall-in",          # 12
]

NODE_MASK_IDX: Final[int] = 13
NODE_PAD_IDX: Final[int] = 14
NODE_VOCAB_SIZE: Final[int] = 15  # 13 room types + MASK + PAD

# ---------------------------------------------------------------------------
# Edge vocabulary (N_MAX-independent)
# ---------------------------------------------------------------------------

EDGE_TYPES: Final[list[str]] = [
    "left-above",       # 0  — inverse: right-below (9)
    "left-below",       # 1  — inverse: right-above (8)
    "left-of",          # 2  — inverse: right-of (7)
    "above",            # 3  — inverse: below (6)
    "inside",           # 4  — inverse: surrounding (5)
    "surrounding",      # 5  — inverse: inside (4)
    "below",            # 6  — inverse: above (3)
    "right-of",         # 7  — inverse: left-of (2)
    "right-above",      # 8  — inverse: left-below (1)
    "right-below",      # 9  — inverse: left-above (0)
]

EDGE_NO_EDGE_IDX: Final[int] = 10
EDGE_MASK_IDX: Final[int] = 11
EDGE_PAD_IDX: Final[int] = 12
EDGE_VOCAB_SIZE: Final[int] = 13  # 10 relationships + no-edge + MASK + PAD

# ---------------------------------------------------------------------------
# VocabConfig: dataset-dependent sizing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VocabConfig:
    """Single source of truth for N_MAX-derived constants.

    Immutable after creation. All sizes (n_edges, seq_len) are derived
    from n_max to prevent mismatches between model and data.

    Args:
        n_max: Maximum number of rooms per graph.
               8 for RPLAN, 14 for ResPlan.
    """

    n_max: int

    @property
    def n_edges(self) -> int:
        """Number of upper-triangle edge positions: C(n_max, 2)."""
        return self.n_max * (self.n_max - 1) // 2

    @property
    def seq_len(self) -> int:
        """Total sequence length: n_max node positions + n_edges edge positions."""
        return self.n_max + self.n_edges

    def compute_pad_mask(self, num_rooms: int) -> Tensor:
        """Compute a boolean mask indicating real (non-PAD) positions.

        Args:
            num_rooms: Number of actual rooms in this graph (1 <= num_rooms <= n_max).

        Returns:
            Tensor of shape (seq_len,), dtype=torch.bool.
            True = real position, False = PAD position.

        The first num_rooms node positions are real; remaining node
        positions are PAD. An edge position (i, j) is real only if
        BOTH i < num_rooms AND j < num_rooms.
        """
        if not (1 <= num_rooms <= self.n_max):
            raise ValueError(
                f"num_rooms must be in [1, {self.n_max}], got {num_rooms}"
            )

        mask = torch.zeros(self.seq_len, dtype=torch.bool)

        # Node positions: first num_rooms are real
        mask[:num_rooms] = True

        # Edge positions: real only if both endpoints are real rooms
        for pos in range(self.n_edges):
            i, j = self.edge_position_to_pair(pos)
            if i < num_rooms and j < num_rooms:
                mask[self.n_max + pos] = True

        return mask

    def edge_position_to_pair(self, pos: int) -> tuple[int, int]:
        """Convert a flat edge position index to a (i, j) node pair.

        Args:
            pos: Edge position index, 0 <= pos < n_edges.

        Returns:
            (i, j) where 0 <= i < j < n_max.

        The mapping uses upper-triangle row-major order:
            pos 0 -> (0, 1), pos 1 -> (0, 2), ...,
            pos n_max-2 -> (0, n_max-1), pos n_max-1 -> (1, 2), ...
        """
        if not (0 <= pos < self.n_edges):
            raise ValueError(
                f"pos must be in [0, {self.n_edges}), got {pos}"
            )

        # Row-major upper triangle enumeration
        row = 0
        cumulative = 0
        while cumulative + (self.n_max - 1 - row) <= pos:
            cumulative += self.n_max - 1 - row
            row += 1
        col = pos - cumulative + row + 1
        return (row, col)

    def pair_to_edge_position(self, i: int, j: int) -> int:
        """Convert a (i, j) node pair to a flat edge position index.

        Args:
            i: First node index, 0 <= i < n_max.
            j: Second node index, 0 <= j < n_max, j != i.

        Returns:
            Edge position index, 0 <= result < n_edges.

        If i > j, the pair is automatically swapped (upper triangle).
        """
        if i == j:
            raise ValueError(f"Self-loops not allowed: i == j == {i}")
        if not (0 <= i < self.n_max and 0 <= j < self.n_max):
            raise ValueError(
                f"Indices must be in [0, {self.n_max}), got ({i}, {j})"
            )

        # Ensure upper triangle: i < j
        if i > j:
            i, j = j, i

        # Position = cumulative entries before row i + offset within row i
        return i * self.n_max - i * (i + 1) // 2 + (j - i - 1)


# ---------------------------------------------------------------------------
# Preset configurations
# ---------------------------------------------------------------------------

RPLAN_VOCAB_CONFIG: Final[VocabConfig] = VocabConfig(n_max=8)
# RPLAN: seq_len = 8 + 28 = 36

RESPLAN_VOCAB_CONFIG: Final[VocabConfig] = VocabConfig(n_max=14)
# ResPlan: seq_len = 14 + 91 = 105
