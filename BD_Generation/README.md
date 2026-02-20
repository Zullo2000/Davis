# BD-Gen: Bubble Diagram Generation via Masked Discrete Diffusion

## Overview

BD-Gen is an MDLM-based discrete diffusion model for generating residential floorplan bubble diagrams. It represents bubble diagrams as flat token sequences (nodes + edges) with PAD handling for variable-size graphs. The model learns to denoise fully masked sequences into valid bubble diagrams that capture room types, spatial relationships, and connectivity constraints found in real residential floorplans.

## Architecture

- **VocabConfig** defines vocabulary sizing and sequence layout parameters.
- Token sequences have length `SEQ_LEN` (36 for RPLAN with `n_max=8`): node tokens followed by edge tokens in upper-triangular order.
- A custom transformer with **adaLN-Zero** time conditioning processes the token sequence.
- Training uses the **MDLM continuous-time ELBO** loss with importance-sampled timesteps and per-sample normalization over active (non-PAD) positions.

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
cd BD_Generation
pip install -e ".[dev]"
```

### Verification

```bash
python -c "from bd_gen.data.vocab import NODE_VOCAB_SIZE, RPLAN_VOCAB_CONFIG; print(NODE_VOCAB_SIZE, RPLAN_VOCAB_CONFIG)"
```

## Project Structure

```
BD_Generation/
    bd_gen/
        __init__.py
        data/
            __init__.py
            vocab.py
            dataset.py
            collate.py
        model/
            __init__.py
            transformer.py
            adaln.py
        diffusion/
            __init__.py
            noise_schedule.py
            mdlm.py
        eval/
            __init__.py
            metrics.py
            validity.py
        viz/
            __init__.py
            graph_plot.py
        utils/
            __init__.py
            helpers.py
    configs/
        config.yaml
        model/
        noise/
        training/
    scripts/
        prepare_data.py
        train.py
        evaluate.py
        sample.py
    tests/
        test_vocab.py
        test_dataset.py
        test_noise.py
        test_model.py
        test_diffusion.py
    notebooks/
    pyproject.toml
    Makefile
    README.md
```

## Running Experiments

Training is managed via Hydra. Examples:

```bash
# Default configuration
python scripts/train.py

# Override model and noise schedule
python scripts/train.py model=base noise=cosine

# Quick local run without W&B logging
python scripts/train.py wandb.mode=disabled training.epochs=5
```

## Testing

```bash
pytest tests/ -v
```

Or using Make:

```bash
make test
```

## Development

- **Branch naming**: `feature/<short-description>`, `fix/<short-description>`, `refactor/<short-description>`
- **Commit messages**: Follow [Conventional Commits](https://www.conventionalcommits.org/) (e.g., `feat: add noise schedule`, `fix: correct PAD masking`, `test: add vocab round-trip tests`).
- See `planning_T1_with_fixed_forward_process.md` for implementation phases and `research_T1.md` for design rationale.

## Note on Makefile

On Windows, `make` requires GNU Make (install via `choco install make` or `scoop install make`). Alternatively, run the commands from the Makefile targets directly in your terminal.
