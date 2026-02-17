"""Integration tests: full training step + checkpoint roundtrip.

Verifies the complete training pipeline works end-to-end using the
synthetic ``sample_batch`` fixture (no real dataset required).
"""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from bd_gen.data.vocab import EDGE_VOCAB_SIZE, RPLAN_VOCAB_CONFIG
from bd_gen.diffusion.forward_process import forward_mask
from bd_gen.diffusion.loss import ELBOLoss
from bd_gen.diffusion.noise_schedule import LinearSchedule
from bd_gen.model.denoiser import BDDenoiser
from bd_gen.utils.checkpoint import load_checkpoint, save_checkpoint
from bd_gen.utils.seed import set_seed

# -------------------------------------------------------------------
# Test 1: One complete training step
# -------------------------------------------------------------------


def test_full_train_step(sample_batch, dummy_model, linear_schedule):
    """Forward mask -> model -> loss -> backward -> step succeeds."""
    set_seed(42)
    vc = RPLAN_VOCAB_CONFIG

    model = dummy_model
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = ELBOLoss(
        edge_class_weights=torch.ones(EDGE_VOCAB_SIZE),
        vocab_config=vc,
    )

    tokens = sample_batch["tokens"]
    pad_mask = sample_batch["pad_mask"]
    B = tokens.size(0)

    t = torch.rand(B)
    t = torch.clamp(t, min=1e-5, max=1.0)

    x_t, mask_indicators = forward_mask(tokens, pad_mask, t, linear_schedule, vc)
    node_logits, edge_logits = model(x_t, pad_mask, t)
    loss = criterion(
        node_logits,
        edge_logits,
        tokens,
        x_t,
        pad_mask,
        mask_indicators,
        t,
        linear_schedule,
    )

    assert loss.dim() == 0, "Loss must be a scalar"
    assert torch.isfinite(loss), f"Loss must be finite, got {loss.item()}"

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Verify gradients flowed to all trainable parameters and are finite.
    # The finiteness check documents the invariant that SUBS zero masking
    # (-inf logits for MASK/PAD) + safe CE targets + loss_mask produce
    # well-defined gradients through the full training pipeline.
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), \
                f"Non-finite gradient for {name}"


# -------------------------------------------------------------------
# Test 2: Loss decreases over multiple optimisation steps
# -------------------------------------------------------------------


def test_loss_decreases_over_steps(sample_batch, dummy_model, linear_schedule):
    """Loss trend: last-10-average < first-10-average over 50 steps.

    Uses a fixed timestep (t=0.5) so the ELBO weight w(t) is constant
    across steps, eliminating the high variance that random t introduces
    with a tiny batch.
    """
    set_seed(123)
    vc = RPLAN_VOCAB_CONFIG

    model = dummy_model
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = ELBOLoss(
        edge_class_weights=torch.ones(EDGE_VOCAB_SIZE),
        vocab_config=vc,
    )

    tokens = sample_batch["tokens"]
    pad_mask = sample_batch["pad_mask"]
    B = tokens.size(0)

    # Fixed t removes variance from ELBO weight w(t)
    t = torch.full((B,), 0.5)

    losses: list[float] = []
    for _ in range(50):
        x_t, mask_indicators = forward_mask(
            tokens, pad_mask, t, linear_schedule, vc,
        )
        node_logits, edge_logits = model(x_t, pad_mask, t)
        loss = criterion(
            node_logits,
            edge_logits,
            tokens,
            x_t,
            pad_mask,
            mask_indicators,
            t,
            linear_schedule,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    first_10_avg = sum(losses[:10]) / 10
    last_10_avg = sum(losses[-10:]) / 10
    assert last_10_avg < first_10_avg, (
        f"Loss did not decrease: first 10 avg={first_10_avg:.4f}, "
        f"last 10 avg={last_10_avg:.4f}"
    )


# -------------------------------------------------------------------
# Test 3: Checkpoint save/load roundtrip
# -------------------------------------------------------------------


def test_checkpoint_roundtrip(sample_batch, tmp_path):
    """Save, load into fresh model, verify identical output."""
    set_seed(42)
    vc = RPLAN_VOCAB_CONFIG

    # Create model and take one optimisation step so weights differ
    # from initialisation.
    model = BDDenoiser(
        d_model=32, n_layers=1, n_heads=2, vocab_config=vc, dropout=0.0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = ELBOLoss(
        edge_class_weights=torch.ones(EDGE_VOCAB_SIZE),
        vocab_config=vc,
    )
    schedule = LinearSchedule()

    tokens = sample_batch["tokens"]
    pad_mask = sample_batch["pad_mask"]
    t = torch.tensor([0.5, 0.5, 0.5, 0.5])

    x_t, mask_ind = forward_mask(tokens, pad_mask, t, schedule, vc)
    node_logits, edge_logits = model(x_t, pad_mask, t)
    loss = criterion(
        node_logits, edge_logits, tokens, x_t, pad_mask, mask_ind, t, schedule,
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Get reference output after the step
    model.eval()
    with torch.no_grad():
        ref_node, ref_edge = model(x_t, pad_mask, t)

    # Save checkpoint
    fake_config = OmegaConf.create({"test": True, "seed": 42})
    ckpt_path = tmp_path / "test_checkpoint.pt"
    save_checkpoint(model, optimizer, epoch=0, config=fake_config, path=ckpt_path)

    # Load into a fresh model
    fresh_model = BDDenoiser(
        d_model=32, n_layers=1, n_heads=2, vocab_config=vc, dropout=0.0,
    )
    fresh_optimizer = torch.optim.AdamW(fresh_model.parameters(), lr=1e-3)
    meta = load_checkpoint(ckpt_path, fresh_model, fresh_optimizer)

    assert meta["epoch"] == 0
    assert meta["config"]["seed"] == 42

    # Verify identical output
    fresh_model.eval()
    with torch.no_grad():
        loaded_node, loaded_edge = fresh_model(x_t, pad_mask, t)

    assert torch.allclose(ref_node, loaded_node, atol=1e-6), (
        "Node logits differ after checkpoint roundtrip"
    )
    assert torch.allclose(ref_edge, loaded_edge, atol=1e-6), (
        "Edge logits differ after checkpoint roundtrip"
    )


# -------------------------------------------------------------------
# Test 4: Seed reproducibility
# -------------------------------------------------------------------


def test_seed_reproducibility():
    """Same seed produces identical random sequences."""
    set_seed(999)
    a = torch.randn(10)

    set_seed(999)
    b = torch.randn(10)

    assert torch.equal(a, b), "set_seed should produce identical sequences"


# -------------------------------------------------------------------
# Test 5: LR warmup schedule
# -------------------------------------------------------------------


def test_lr_warmup_schedule():
    """Linear warmup reaches target LR at warmup_steps, stays constant after."""
    model = torch.nn.Linear(10, 10)
    base_lr = 3e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)

    warmup_steps = 100

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # At step 0: lr = 0
    assert scheduler.get_last_lr()[0] == pytest.approx(0.0, abs=1e-8)

    # Step to halfway through warmup
    for _ in range(50):
        scheduler.step()
    assert scheduler.get_last_lr()[0] == pytest.approx(
        base_lr * 50 / 100, rel=1e-5,
    )

    # Step to the end of warmup
    for _ in range(50):
        scheduler.step()
    assert scheduler.get_last_lr()[0] == pytest.approx(base_lr, rel=1e-5)

    # Beyond warmup: stays constant
    for _ in range(200):
        scheduler.step()
    assert scheduler.get_last_lr()[0] == pytest.approx(base_lr, rel=1e-5)
