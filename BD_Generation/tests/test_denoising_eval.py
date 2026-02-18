"""Tests for bd_gen.eval.denoising_eval."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset

from bd_gen.data.vocab import VocabConfig
from omegaconf import OmegaConf

from bd_gen.diffusion.noise_schedule import get_noise
from bd_gen.eval.denoising_eval import denoising_eval


class _PerfectDenoiser(torch.nn.Module):
    """Mock denoiser that returns one-hot logits for ground truth tokens."""

    def __init__(self, x0: torch.Tensor, vocab_config: VocabConfig):
        super().__init__()
        self.x0 = x0  # (B, seq_len)
        self.vocab_config = vocab_config
        from bd_gen.data.vocab import NODE_VOCAB_SIZE, EDGE_VOCAB_SIZE

        self.node_vocab = NODE_VOCAB_SIZE
        self.edge_vocab = EDGE_VOCAB_SIZE

    def forward(self, tokens, pad_mask, t, condition=None):
        B = tokens.size(0)
        n_max = self.vocab_config.n_max

        # Perfect node logits: high value at ground truth index
        node_logits = torch.full(
            (B, n_max, self.node_vocab), -100.0, device=tokens.device,
        )
        for b in range(B):
            for i in range(n_max):
                node_logits[b, i, self.x0[b, i]] = 100.0

        # Perfect edge logits
        n_edges = tokens.size(1) - n_max
        edge_logits = torch.full(
            (B, n_edges, self.edge_vocab), -100.0, device=tokens.device,
        )
        for b in range(B):
            for i in range(n_edges):
                edge_logits[b, i, self.x0[b, n_max + i]] = 100.0

        return node_logits, edge_logits


def _make_val_dataloader(vocab_config: VocabConfig, n_samples: int = 8):
    """Build a tiny synthetic validation dataloader."""
    seq_len = vocab_config.seq_len
    from bd_gen.data.vocab import NODE_PAD_IDX, EDGE_PAD_IDX

    tokens_list = []
    pad_masks_list = []
    for _ in range(n_samples):
        num_rooms = 4
        pad_mask = vocab_config.compute_pad_mask(num_rooms)
        tokens = torch.zeros(seq_len, dtype=torch.long)
        # Fill nodes with valid types (0..12)
        for i in range(num_rooms):
            tokens[i] = i % 13
        # Fill remaining nodes with PAD
        for i in range(num_rooms, vocab_config.n_max):
            tokens[i] = NODE_PAD_IDX
        # Fill edges: active edges get type 2 (left-of), rest PAD
        n_max = vocab_config.n_max
        for idx in range(n_max, seq_len):
            if pad_mask[idx]:
                tokens[idx] = 2
            else:
                tokens[idx] = EDGE_PAD_IDX
        tokens_list.append(tokens)
        pad_masks_list.append(pad_mask)

    all_tokens = torch.stack(tokens_list)
    all_pads = torch.stack(pad_masks_list)

    class _DictDataset:
        def __init__(self, tokens, pad_masks):
            self.tokens = tokens
            self.pad_masks = pad_masks

        def __len__(self):
            return self.tokens.size(0)

        def __getitem__(self, idx):
            return {"tokens": self.tokens[idx], "pad_mask": self.pad_masks[idx]}

    ds = _DictDataset(all_tokens, all_pads)
    return DataLoader(ds, batch_size=4, shuffle=False), all_tokens


class TestDenoisingEval:
    def test_returns_keys_for_each_t(self):
        vocab_config = VocabConfig(n_max=8)
        loader, x0 = _make_val_dataloader(vocab_config)
        noise = get_noise(OmegaConf.create({"type": "cosine", "eps": 1e-3}))
        model = _PerfectDenoiser(x0, vocab_config)
        model.eval()

        t_grid = [0.3, 0.7]
        result = denoising_eval(
            model, loader, noise, vocab_config,
            t_grid=t_grid, device="cpu", max_batches=2,
        )

        for t_val in t_grid:
            t_str = f"{t_val:.1f}"
            assert f"denoise/acc_node@t={t_str}" in result
            assert f"denoise/acc_edge@t={t_str}" in result
            assert f"denoise/ce_node@t={t_str}" in result
            assert f"denoise/ce_edge@t={t_str}" in result

    def test_perfect_model_high_accuracy(self):
        vocab_config = VocabConfig(n_max=8)
        loader, x0 = _make_val_dataloader(vocab_config)
        noise = get_noise(OmegaConf.create({"type": "cosine", "eps": 1e-3}))
        model = _PerfectDenoiser(x0, vocab_config)
        model.eval()

        result = denoising_eval(
            model, loader, noise, vocab_config,
            t_grid=[0.5], device="cpu", max_batches=2,
        )

        # Perfect denoiser should achieve ~1.0 accuracy
        assert result["denoise/acc_node@t=0.5"] > 0.99
        assert result["denoise/acc_edge@t=0.5"] > 0.99

    def test_pad_positions_excluded(self):
        """Verify that PAD positions don't contribute to accuracy counts."""
        vocab_config = VocabConfig(n_max=8)
        loader, x0 = _make_val_dataloader(vocab_config, n_samples=4)
        noise = get_noise(OmegaConf.create({"type": "cosine", "eps": 1e-3}))
        model = _PerfectDenoiser(x0, vocab_config)
        model.eval()

        # At t=0.9, most positions are masked. Only non-PAD masked positions
        # should be scored. The perfect model gets them all right.
        result = denoising_eval(
            model, loader, noise, vocab_config,
            t_grid=[0.9], device="cpu", max_batches=1,
        )
        assert result["denoise/acc_node@t=0.9"] > 0.99

    def test_max_batches_limits_evaluation(self):
        vocab_config = VocabConfig(n_max=8)
        loader, x0 = _make_val_dataloader(vocab_config, n_samples=16)
        noise = get_noise(OmegaConf.create({"type": "cosine", "eps": 1e-3}))
        model = _PerfectDenoiser(x0, vocab_config)
        model.eval()

        # With max_batches=1, should only process 4 samples (batch_size=4)
        result = denoising_eval(
            model, loader, noise, vocab_config,
            t_grid=[0.5], device="cpu", max_batches=1,
        )
        assert "denoise/acc_node@t=0.5" in result
