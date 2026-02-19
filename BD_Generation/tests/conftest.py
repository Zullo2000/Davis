"""Shared pytest fixtures for bd_gen test suite.

Fixtures defined here are automatically available to all test files.
"""

import pytest
import torch

from bd_gen.data.vocab import (
    EDGE_MASK_IDX,
    EDGE_NO_EDGE_IDX,
    EDGE_PAD_IDX,
    EDGE_TYPES,
    EDGE_VOCAB_SIZE,
    NODE_MASK_IDX,
    NODE_PAD_IDX,
    NODE_TYPES,
    NODE_VOCAB_SIZE,
    RPLAN_VOCAB_CONFIG,
    VocabConfig,
)


@pytest.fixture
def vocab_config() -> VocabConfig:
    """Return the RPLAN VocabConfig preset."""
    return RPLAN_VOCAB_CONFIG


@pytest.fixture
def vocab_constants() -> dict:
    """Return all vocabulary constants as a dict for easy assertion."""
    return {
        "NODE_TYPES": NODE_TYPES,
        "EDGE_TYPES": EDGE_TYPES,
        "NODE_MASK_IDX": NODE_MASK_IDX,
        "NODE_PAD_IDX": NODE_PAD_IDX,
        "NODE_VOCAB_SIZE": NODE_VOCAB_SIZE,
        "EDGE_NO_EDGE_IDX": EDGE_NO_EDGE_IDX,
        "EDGE_MASK_IDX": EDGE_MASK_IDX,
        "EDGE_PAD_IDX": EDGE_PAD_IDX,
        "EDGE_VOCAB_SIZE": EDGE_VOCAB_SIZE,
    }


@pytest.fixture
def sample_batch() -> dict:
    """Create a batch of 4 samples with varying num_rooms.

    Returns a dict with:
        "tokens": Tensor(4, 36, dtype=torch.long)
        "pad_mask": Tensor(4, 36, dtype=torch.bool)
        "num_rooms": [2, 4, 6, 8]

    Tokens contain valid vocab indices with correct PAD placement:
    - Node PAD positions -> NODE_PAD_IDX (14)
    - Edge PAD positions -> EDGE_PAD_IDX (12)
    - Real node positions -> random in [0, 12] (valid room types)
    - Real edge positions -> random in [0, 10] (valid edge types incl. no-edge)
    """
    vc = RPLAN_VOCAB_CONFIG
    num_rooms_list = [2, 4, 6, 8]
    batch_size = len(num_rooms_list)

    tokens = torch.zeros(batch_size, vc.seq_len, dtype=torch.long)
    pad_mask = torch.zeros(batch_size, vc.seq_len, dtype=torch.bool)

    torch.manual_seed(42)  # reproducible fixture

    for b, num_rooms in enumerate(num_rooms_list):
        mask = vc.compute_pad_mask(num_rooms)
        pad_mask[b] = mask

        # Fill node positions
        for k in range(vc.n_max):
            if k < num_rooms:
                tokens[b, k] = torch.randint(0, len(NODE_TYPES), (1,))
            else:
                tokens[b, k] = NODE_PAD_IDX

        # Fill edge positions
        for pos in range(vc.n_edges):
            seq_idx = vc.n_max + pos
            if mask[seq_idx]:
                # Real edge: random valid type (0-9) or no-edge (10)
                tokens[b, seq_idx] = torch.randint(0, EDGE_NO_EDGE_IDX + 1, (1,))
            else:
                tokens[b, seq_idx] = EDGE_PAD_IDX

    return {
        "tokens": tokens,
        "pad_mask": pad_mask,
        "num_rooms": num_rooms_list,
    }


@pytest.fixture
def dummy_model():
    """Small BDDenoiser for fast test execution.

    Uses d_model=32, n_layers=1, n_heads=2 with RPLAN_VOCAB_CONFIG.
    Dropout disabled for deterministic test behavior.
    """
    from bd_gen.model.denoiser import BDDenoiser

    return BDDenoiser(
        d_model=32,
        n_layers=1,
        n_heads=2,
        vocab_config=RPLAN_VOCAB_CONFIG,
        dropout=0.0,
    )


# ---------------------------------------------------------------------------
# Diffusion fixtures (Phase 3+)
# ---------------------------------------------------------------------------


@pytest.fixture
def linear_schedule():
    """LinearSchedule with default parameters (sigma_min=0, sigma_max=10)."""
    from bd_gen.diffusion.noise_schedule import LinearSchedule

    return LinearSchedule(sigma_min=0.0, sigma_max=10.0)


@pytest.fixture
def loglinear_schedule():
    """LogLinearSchedule with default parameters (eps=1e-3)."""
    from bd_gen.diffusion.noise_schedule import LogLinearSchedule

    return LogLinearSchedule(eps=1e-3)


@pytest.fixture
def cosine_schedule():
    """CosineSchedule with default parameters (eps=1e-3)."""
    from bd_gen.diffusion.noise_schedule import CosineSchedule

    return CosineSchedule(eps=1e-3)


@pytest.fixture
def edge_class_weights():
    """Uniform edge class weights for testing (all 1.0)."""
    return torch.ones(EDGE_VOCAB_SIZE, dtype=torch.float32)


@pytest.fixture
def elbo_loss(edge_class_weights, vocab_config):
    """ELBOLoss with uniform weights for testing."""
    from bd_gen.diffusion.loss import ELBOLoss

    return ELBOLoss(
        edge_class_weights=edge_class_weights,
        node_class_weights=None,
        vocab_config=vocab_config,
    )
