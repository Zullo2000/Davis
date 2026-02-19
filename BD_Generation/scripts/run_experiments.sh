#!/bin/bash
# =============================================================================
# ReMDM Remasking Experiment Suite
# =============================================================================
#
# Runs the full layered experiment design from remasking_design.md Section 10:
#   - Retrain with post-v1 improvements (SUBS zero masking, float64 ELBO, IS)
#   - Layer 1: 4 baselines (random/llada x argmax/top-p, no remasking)
#   - Layer 2: 5 cap eta sweep runs (top-p=0.9, random unmasking)
#   - Layer 3: 3 confidence + t_switch sweep runs (top-p=0.9, random unmasking)
#   - Generate comparison table
#
# Run 10 (cap best_eta + argmax control) is deferred — run manually after
# reviewing Layer 2 results to pick the best eta.
#
# If Layer 1 shows llada > random, re-run Layers 2-3 with
# eval.unmasking_mode=llada.
#
# Usage:
#   cd BD_Generation
#   bash scripts/run_experiments.sh
#
# Estimated time: ~70-90 min on RTX A5000
# =============================================================================

set -euo pipefail

# --- Configuration ---
EVAL_CMD="python scripts/evaluate.py"
TRAIN_CMD="python scripts/train.py"
COMMON="wandb.mode=disabled"

echo "============================================"
echo "  ReMDM Experiment Suite"
echo "  Started: $(date)"
echo "============================================"

# =============================================================================
# Phase 0: Retrain
# =============================================================================
echo ""
echo "=== Phase 0: Retrain with post-v1 improvements ==="
echo "  Changes: SUBS zero masking, float64 ELBO, importance sampling"
echo ""

$TRAIN_CMD $COMMON

# Find the latest checkpoint
LATEST_OUTPUT=$(ls -td outputs/*/checkpoints/checkpoint_final.pt 2>/dev/null | head -1)
if [ -z "$LATEST_OUTPUT" ]; then
    echo "ERROR: No checkpoint found after training!"
    exit 1
fi
echo "Checkpoint: $LATEST_OUTPUT"

CKPT="eval.checkpoint_path=$LATEST_OUTPUT"

# =============================================================================
# Phase 1: Layer 1 — Baselines (4 runs, no remasking)
# =============================================================================
echo ""
echo "=== Phase 1: Layer 1 — Baselines (4 runs) ==="
echo ""

# Run 1: random + argmax + no remasking
echo "--- Run 1/12: random + argmax ---"
$EVAL_CMD $COMMON $CKPT \
    eval.unmasking_mode=random eval.temperature=0.0 eval.top_p=null \
    eval.remasking.enabled=false

# Run 2: random + top-p=0.9 + no remasking
echo "--- Run 2/12: random + top-p=0.9 ---"
$EVAL_CMD $COMMON $CKPT \
    eval.unmasking_mode=random eval.top_p=0.9 \
    eval.remasking.enabled=false

# Run 3: llada + argmax + no remasking
echo "--- Run 3/12: llada + argmax ---"
$EVAL_CMD $COMMON $CKPT \
    eval.unmasking_mode=llada eval.temperature=0.0 eval.top_p=null \
    eval.remasking.enabled=false

# Run 4: llada + top-p=0.9 + no remasking
echo "--- Run 4/12: llada + top-p=0.9 ---"
$EVAL_CMD $COMMON $CKPT \
    eval.unmasking_mode=llada eval.top_p=0.9 \
    eval.remasking.enabled=false

echo ""
echo "Layer 1 complete. Review eval_results/ to pick winner unmasking mode."
echo "Continuing with random unmasking for Layers 2-3."
echo ""

# =============================================================================
# Phase 2: Layer 2 — Cap Eta Sweep (5 runs)
# =============================================================================
echo "=== Phase 2: Layer 2 — Cap eta sweep (5 runs) ==="
echo "  Fixed: random unmasking, top-p=0.9, t_switch=1.0"
echo ""

RUN_NUM=5
for ETA in 0.2 0.4 0.6 0.8 1.0; do
    echo "--- Run $RUN_NUM/12: cap eta=$ETA ---"
    $EVAL_CMD $COMMON $CKPT \
        eval.unmasking_mode=random eval.top_p=0.9 \
        eval.remasking.enabled=true eval.remasking.strategy=cap \
        eval.remasking.eta=$ETA eval.remasking.t_switch=1.0
    RUN_NUM=$((RUN_NUM + 1))
done

echo ""
echo "Layer 2 complete."
echo "Run 10 (cap best_eta + argmax) deferred — run manually after analysis."
echo ""

# =============================================================================
# Phase 3: Layer 3 — Confidence + Switch (3 runs)
# =============================================================================
echo "=== Phase 3: Layer 3 — Confidence + t_switch sweep (3 runs) ==="
echo "  Fixed: random unmasking, top-p=0.9, confidence strategy (no eta)"
echo ""

RUN_NUM=11
for TSW in 0.3 0.5 0.7; do
    echo "--- Run $RUN_NUM/12: confidence t_switch=$TSW ---"
    $EVAL_CMD $COMMON $CKPT \
        eval.unmasking_mode=random eval.top_p=0.9 \
        eval.remasking.enabled=true eval.remasking.strategy=confidence \
        eval.remasking.eta=0.0 eval.remasking.t_switch=$TSW
    RUN_NUM=$((RUN_NUM + 1))
done

echo ""
echo "Layer 3 complete."
echo ""

# =============================================================================
# Phase 4: Generate Comparison Table
# =============================================================================
echo "=== Phase 4: Generate comparison table ==="
python scripts/compare.py
echo ""

# =============================================================================
# Done
# =============================================================================
echo "============================================"
echo "  All 12 runs complete!"
echo "  Finished: $(date)"
echo "============================================"
echo ""
echo "Results:"
echo "  - JSON files: eval_results/*.json"
echo "  - Comparison: eval_results/comparison.md"
echo ""
echo "Next steps:"
echo "  1. Review Layer 1 baselines — if llada wins, re-run L2/L3 with llada"
echo "  2. Review Layer 2 — pick best eta, run argmax control (Run 10):"
echo "     $EVAL_CMD $COMMON $CKPT \\"
echo "       eval.unmasking_mode=random eval.temperature=0.0 eval.top_p=null \\"
echo "       eval.remasking.enabled=true eval.remasking.strategy=cap \\"
echo "       eval.remasking.eta=<BEST> eval.remasking.t_switch=1.0"
echo "  3. Layer 4: compare best cap vs best confidence from comparison.md"
