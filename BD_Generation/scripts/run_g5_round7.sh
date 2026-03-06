#!/bin/bash
# =============================================================================
# G5 Round 7: Decay Remasking Schedule (p=3) — Option A vs Option B
# =============================================================================
#
# Purpose: Test a power-law decay on sigma_max: sigma_max_eff = sigma_max * t^p.
# This tapers remasking smoothly instead of using an abrupt EMA lock.
# With p=3 and 94 active positions:
#   - Step 30 (t=0.70): ~32 tokens remasked
#   - Step 50 (t=0.50): ~12 tokens remasked
#   - Step 65 (t=0.35): ~2 tokens remasked
#   - Step 80 (t=0.20): ~0 tokens remasked
#
# No EMA lock — the decay schedule replaces it entirely.
#
# Grid:  2 configs:
#   1. confidence + Option A (fresh logits) + decay p=3 (no lock)
#   2. confidence + Option B (protect just-unmasked) + decay p=3 (no lock)
#
# Fixed: K=16, alpha=0.01, v1 loglinear checkpoint, soft reward only
# Seeds: [42, 123, 456] (3 seeds), 200 samples/seed = 600 per config
#
# Results saved to: eval_results/loglinear_noise_sc/round7_guid/
#
# Usage:
#   cd BD_Generation
#   bash scripts/run_g5_round7.sh calibrate    # Step 1: calibrate (CPU, ~30s)
#   bash scripts/run_g5_round7.sh generate     # Step 2: generate 2 configs (GPU)
#   bash scripts/run_g5_round7.sh evaluate     # Step 3: evaluate (CPU)
#   bash scripts/run_g5_round7.sh compare      # Step 4: comparison table (CPU)
#   bash scripts/run_g5_round7.sh analyze      # Step 5: analysis plots (CPU)
#   bash scripts/run_g5_round7.sh organize     # Step 6: move results to round7_guid/
#   bash scripts/run_g5_round7.sh all          # Steps 1-6 sequentially
# =============================================================================

set -euo pipefail

# --- Configuration ---
GUIDANCE_CONFIG="configs/guidance/example_basic.yaml"
COMMON="wandb.mode=disabled"

V1_CKPT="outputs/2026-02-19_16-58-23/checkpoints/checkpoint_final.pt"
CAL_FILE_CONFIDENCE="configs/guidance/calibration_v1_confidence.json"

# Shared Hydra overrides
HYDRA_COMMON="noise=loglinear eval.unmasking_mode=llada eval.top_p=0.9"

# Variant-specific Hydra overrides
HYDRA_CONFIDENCE="eval.remasking.enabled=true eval.remasking.strategy=confidence eval.remasking.t_switch=1.0"

# 3 seeds x 200 samples = 600 per config
HYDRA_SAMPLES="eval.seeds=[42,123,456] eval.num_samples=200"

# Fixed hyperparameters
K=16
ALPHA=0.01

# Decay schedule parameter
DECAY_POWER=3

# Baselines (from prior rounds, already generated)
BASELINE_NOREMASK="llada_topp0.9_no_remask"
BASELINE_CONFIDENCE="llada_topp0.9_remdm_confidence_tsw1.0"

# Round 7 models
R7_OPTA="${BASELINE_CONFIDENCE}_guided_r7decayA_K${K}_a${ALPHA}"
R7_OPTB="${BASELINE_CONFIDENCE}_guided_r7decayB_K${K}_a${ALPHA}"

# Output
SCHEDULE="loglinear_noise_sc"
EVAL_DIR="eval_results/${SCHEDULE}"
ROUND7_DIR="${EVAL_DIR}/round7_guid"
COMPARISON_FILE="${ROUND7_DIR}/comparison_guided_round7.md"

# =============================================================================

print_header() {
    echo ""
    echo "============================================"
    echo "  $1"
    echo "  $(date)"
    echo "============================================"
}

# --- Step 1: Calibrate (reuse prior calibration if available) ---
step_calibrate() {
    print_header "Calibrate P90 normalizers (confidence variant)"

    if [ -f "$CAL_FILE_CONFIDENCE" ]; then
        echo "Calibration file already exists: $CAL_FILE_CONFIDENCE"
        cat "$CAL_FILE_CONFIDENCE"
        echo ""
        echo "Skipping calibration (reusing prior calibration)."
    else
        echo "--- Calibrating: confidence baseline ---"
        python scripts/calibrate_constraints.py \
            --schedule "$SCHEDULE" \
            --model "$BASELINE_CONFIDENCE" \
            --constraints "$GUIDANCE_CONFIG" \
            --output "$CAL_FILE_CONFIDENCE"
        echo "Saved: $CAL_FILE_CONFIDENCE"
        cat "$CAL_FILE_CONFIDENCE"
    fi
}

# --- Step 2: Generate 2 guided configs with decay schedule ---
step_generate() {
    print_header "Generate 2 guided configs with decay p=$DECAY_POWER (GPU)"
    echo "  K=$K, alpha=$ALPHA, soft reward only"
    echo "  Decay power: $DECAY_POWER (no EMA lock)"
    echo "  Config 1: confidence + Option B (protect just-unmasked) + decay"
    echo "  Config 2: confidence + Option A (fresh logits) + decay"
    echo "  Seeds: [42, 123, 456], 200 samples/seed"

    # --- Config 1: confidence + Option B + decay (runs first — cheaper, evaluate while A runs) ---
    echo ""
    echo "--- Run 1/2: confidence + Option B + decay p=$DECAY_POWER ---"
    echo "    Started: $(date)"
    python scripts/generate_guided.py \
        eval.checkpoint_path="$V1_CKPT" \
        $HYDRA_COMMON \
        $HYDRA_CONFIDENCE \
        $HYDRA_SAMPLES \
        $COMMON \
        --guidance-config "$GUIDANCE_CONFIG" \
        --calibration "$CAL_FILE_CONFIDENCE" \
        --alpha "$ALPHA" --K "$K" \
        --reward-mode soft \
        --protect-just-unmasked \
        --remask-decay-power "$DECAY_POWER" \
        --guidance-tag "r7decayB"
    echo "    Finished: $(date)"

    # --- Config 2: confidence + Option A + decay (2x model cost) ---
    echo ""
    echo "--- Run 2/2: confidence + Option A + decay p=$DECAY_POWER ---"
    echo "    Started: $(date)"
    python scripts/generate_guided.py \
        eval.checkpoint_path="$V1_CKPT" \
        $HYDRA_COMMON \
        $HYDRA_CONFIDENCE \
        $HYDRA_SAMPLES \
        $COMMON \
        --guidance-config "$GUIDANCE_CONFIG" \
        --calibration "$CAL_FILE_CONFIDENCE" \
        --alpha "$ALPHA" --K "$K" \
        --reward-mode soft \
        --fresh-logits-remask \
        --remask-decay-power "$DECAY_POWER" \
        --guidance-tag "r7decayA"
    echo "    Finished: $(date)"

    echo ""
    echo "Generation complete: 2 runs."
}

# --- Step 3: Evaluate (2 new guided models) ---
step_evaluate() {
    print_header "Evaluate 2 new guided models (CPU)"

    local r7_models=("$R7_OPTA" "$R7_OPTB")

    for model in "${r7_models[@]}"; do
        echo ""
        echo "--- Evaluating: $model ---"
        python scripts/evaluate.py \
            --schedule "$SCHEDULE" \
            --model "$model" \
            --guidance-config "$GUIDANCE_CONFIG"
    done

    echo ""
    echo "Evaluation complete."
}

# --- Step 4: Comparison table ---
step_compare() {
    print_header "Generate comparison table (baselines + Round 7)"

    # Create output directory
    mkdir -p "$ROUND7_DIR"

    local models=(
        "$BASELINE_NOREMASK"
        "$BASELINE_CONFIDENCE"
        "$R7_OPTA"
        "$R7_OPTB"
    )

    echo "Comparing ${#models[@]} models:"
    for m in "${models[@]}"; do echo "  - $m"; done

    python scripts/compare_selected.py \
        --schedule "$SCHEDULE" \
        --models "${models[@]}" \
        --guided \
        --output "$COMPARISON_FILE"

    echo ""
    echo "Comparison table: $COMPARISON_FILE"
}

# --- Step 5: Outlier-aware analysis ---
step_analyze() {
    print_header "Outlier-aware analysis plots"

    local r7_models=("$R7_OPTA" "$R7_OPTB")

    for model in "${r7_models[@]}"; do
        echo ""
        echo "--- Analyzing: $model ---"
        python scripts/analyze_guidance_stats.py \
            --schedule "$SCHEDULE" \
            --model "$model" \
            --plot-analysis
    done

    echo ""
    echo "Analysis complete. Check ${EVAL_DIR}/ for *_trajectories_*.png"
}

# --- Step 6: Organize results into round7_guid/ ---
step_organize() {
    print_header "Organize Round 7 results into ${ROUND7_DIR}/"

    mkdir -p "$ROUND7_DIR"

    # Move Round 7 samples, metrics, and plots
    for model in "$R7_OPTA" "$R7_OPTB"; do
        for ext in _samples.pt .json; do
            src="${EVAL_DIR}/${model}${ext}"
            if [ -f "$src" ]; then
                mv "$src" "$ROUND7_DIR/"
                echo "  Moved: $(basename "$src")"
            fi
        done
        # Move analysis PNGs
        for png in "${EVAL_DIR}/${model}"_trajectories_*.png; do
            if [ -f "$png" ]; then
                mv "$png" "$ROUND7_DIR/"
                echo "  Moved: $(basename "$png")"
            fi
        done
        for png in "${EVAL_DIR}/${model}"_analysis_*.png; do
            if [ -f "$png" ]; then
                mv "$png" "$ROUND7_DIR/"
                echo "  Moved: $(basename "$png")"
            fi
        done
    done

    echo ""
    echo "Results organized in: $ROUND7_DIR/"
    ls -la "$ROUND7_DIR/" 2>/dev/null || true
}

# --- Main dispatcher ---
STEP="${1:-help}"

case "$STEP" in
    calibrate)
        step_calibrate
        ;;
    generate)
        step_generate
        ;;
    evaluate)
        step_evaluate
        ;;
    compare)
        step_compare
        ;;
    analyze)
        step_analyze
        ;;
    organize)
        step_organize
        ;;
    all)
        step_calibrate
        step_generate
        step_evaluate
        step_compare
        step_analyze
        step_organize
        ;;
    *)
        echo "G5 Round 7: Decay Remasking Schedule (p=$DECAY_POWER) — Option A vs Option B"
        echo ""
        echo "Usage: bash scripts/run_g5_round7.sh <step>"
        echo ""
        echo "Steps:"
        echo "  calibrate   Reuse/create P90 normalizers (CPU, ~30s)"
        echo "  generate    Generate 2 guided configs with decay schedule (GPU)"
        echo "  evaluate    Evaluate 2 new guided models (CPU)"
        echo "  compare     Comparison table: baselines + R7 (CPU)"
        echo "  analyze     Outlier-aware analysis plots (CPU)"
        echo "  organize    Move Round 7 results to ${ROUND7_DIR}/"
        echo "  all         Run all steps sequentially"
        echo ""
        echo "Grid: 2 configs (Option A + decay p=$DECAY_POWER, Option B + decay p=$DECAY_POWER)"
        echo "Fixed: K=$K, alpha=$ALPHA, soft reward, v1 loglinear"
        echo "Decay: sigma_max *= t^$DECAY_POWER (no EMA lock)"
        echo "Seeds: [42, 123, 456], 200 samples/seed = 600 per config"
        echo ""
        echo "Compares against prior results:"
        echo "  - no-remask + soft                    = 69% (Round 3)"
        echo "  - Option B + lock (no warmup)         = 68.2% (Round 6)"
        echo "  - Option B + lock (warmup=0.2)        = 68.2% (Round 6b)"
        ;;
esac
