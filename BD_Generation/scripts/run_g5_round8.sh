#!/bin/bash
# =============================================================================
# G5 Round 8: K=50 with Vectorized Scoring — Option B variants
# =============================================================================
#
# Purpose: Test whether K=50 improves constraint satisfaction beyond K=16.
# Round 2 K* sweep showed curve still climbing at K=24 for confidence remasking.
# no_bath_kitchen (ForbidAdj) was the bottleneck — kept improving with K.
# Vectorized scoring makes K=50 nearly free (compute ~independent of K).
#
# Hypothesis: K=16 has too high variance for SVDD reweighting.
# K=50 gives more candidates per step, reducing variance.
#
# Grid:  2 configs:
#   1. confidence + Option B (protect just-unmasked), no decay, no lock, K=50
#   2. confidence + Option B (protect just-unmasked) + decay p=3, K=50
#
# Fixed: alpha=0.01, v1 loglinear checkpoint, soft reward only
# Seeds: [42, 123, 456] (3 seeds), 200 samples/seed = 600 per config
#
# Results saved to: eval_results/loglinear_noise_sc/round8_guid/
#
# Usage:
#   cd BD_Generation
#   bash scripts/run_g5_round8.sh calibrate    # Step 1: calibrate (CPU, ~30s)
#   bash scripts/run_g5_round8.sh generate     # Step 2: generate 2 configs (GPU)
#   bash scripts/run_g5_round8.sh evaluate     # Step 3: evaluate (CPU)
#   bash scripts/run_g5_round8.sh compare      # Step 4: comparison table (CPU)
#   bash scripts/run_g5_round8.sh analyze      # Step 5: analysis plots (CPU)
#   bash scripts/run_g5_round8.sh organize     # Step 6: move results to round8_guid/
#   bash scripts/run_g5_round8.sh all          # Steps 1-6 sequentially
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
K=50
ALPHA=0.01

# Decay schedule parameter
DECAY_POWER=3

# Baselines (from prior rounds, already generated)
BASELINE_NOREMASK="llada_topp0.9_no_remask"
BASELINE_CONFIDENCE="llada_topp0.9_remdm_confidence_tsw1.0"

# Round 8 models
R8_OPTB="${BASELINE_CONFIDENCE}_guided_r8optB_K${K}_a${ALPHA}"
R8_OPTB_DECAY="${BASELINE_CONFIDENCE}_guided_r8decayB_K${K}_a${ALPHA}"

# Output
SCHEDULE="loglinear_noise_sc"
EVAL_DIR="eval_results/${SCHEDULE}"
ROUND8_DIR="${EVAL_DIR}/round8_guid"
COMPARISON_FILE="${ROUND8_DIR}/comparison_guided_round8.md"

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

# --- Step 2: Generate 2 guided configs ---
step_generate() {
    print_header "Generate 2 guided configs at K=$K (GPU)"
    echo "  alpha=$ALPHA, soft reward only"
    echo "  Config 1: confidence + Option B, no decay, no lock, K=$K"
    echo "  Config 2: confidence + Option B + decay p=$DECAY_POWER, K=$K"
    echo "  Seeds: [42, 123, 456], 200 samples/seed"

    # --- Config 1: confidence + Option B, no decay, no lock ---
    echo ""
    echo "--- Run 1/2: confidence + Option B, K=$K (no decay, no lock) ---"
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
        --guidance-tag "r8optB"
    echo "    Finished: $(date)"

    # --- Config 2: confidence + Option B + decay p=3 ---
    echo ""
    echo "--- Run 2/2: confidence + Option B + decay p=$DECAY_POWER, K=$K ---"
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
        --guidance-tag "r8decayB"
    echo "    Finished: $(date)"

    echo ""
    echo "Generation complete: 2 runs."
}

# --- Step 3: Evaluate ---
step_evaluate() {
    print_header "Evaluate 2 new guided models (CPU)"

    local r8_models=("$R8_OPTB" "$R8_OPTB_DECAY")

    for model in "${r8_models[@]}"; do
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
    print_header "Generate comparison table (baselines + Round 8)"

    # Create output directory
    mkdir -p "$ROUND8_DIR"

    local models=(
        "$BASELINE_NOREMASK"
        "$BASELINE_CONFIDENCE"
        "$R8_OPTB"
        "$R8_OPTB_DECAY"
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

    local r8_models=("$R8_OPTB" "$R8_OPTB_DECAY")

    for model in "${r8_models[@]}"; do
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

# --- Step 6: Organize results into round8_guid/ ---
step_organize() {
    print_header "Organize Round 8 results into ${ROUND8_DIR}/"

    mkdir -p "$ROUND8_DIR"

    # Move Round 8 samples, metrics, and plots
    for model in "$R8_OPTB" "$R8_OPTB_DECAY"; do
        for ext in _samples.pt .json; do
            src="${EVAL_DIR}/${model}${ext}"
            if [ -f "$src" ]; then
                mv "$src" "$ROUND8_DIR/"
                echo "  Moved: $(basename "$src")"
            fi
        done
        # Move analysis PNGs
        for png in "${EVAL_DIR}/${model}"_trajectories_*.png; do
            if [ -f "$png" ]; then
                mv "$png" "$ROUND8_DIR/"
                echo "  Moved: $(basename "$png")"
            fi
        done
        for png in "${EVAL_DIR}/${model}"_analysis_*.png; do
            if [ -f "$png" ]; then
                mv "$png" "$ROUND8_DIR/"
                echo "  Moved: $(basename "$png")"
            fi
        done
    done

    echo ""
    echo "Results organized in: $ROUND8_DIR/"
    ls -la "$ROUND8_DIR/" 2>/dev/null || true
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
        echo "G5 Round 8: K=50 with Vectorized Scoring — Option B variants"
        echo ""
        echo "Usage: bash scripts/run_g5_round8.sh <step>"
        echo ""
        echo "Steps:"
        echo "  calibrate   Reuse/create P90 normalizers (CPU, ~30s)"
        echo "  generate    Generate 2 guided configs at K=$K (GPU)"
        echo "  evaluate    Evaluate 2 new guided models (CPU)"
        echo "  compare     Comparison table: baselines + R8 (CPU)"
        echo "  analyze     Outlier-aware analysis plots (CPU)"
        echo "  organize    Move Round 8 results to ${ROUND8_DIR}/"
        echo "  all         Run all steps sequentially"
        echo ""
        echo "Grid: 2 configs (both Option B, confidence remasking, K=$K)"
        echo "  1. Option B, no decay, no lock    → pure K=50 effect"
        echo "  2. Option B + decay p=$DECAY_POWER          → K=50 + smooth taper"
        echo ""
        echo "Fixed: alpha=$ALPHA, soft reward, v1 loglinear"
        echo "Seeds: [42, 123, 456], 200 samples/seed = 600 per config"
        echo ""
        echo "Key comparisons:"
        echo "  - R8 Option B K=50     vs  R5 Option B K=16 (56%)  → does K=50 help?"
        echo "  - R8 decay B K=50      vs  R7 decay B K=16         → K=50 + decay?"
        echo "  - R8 Option B K=50     vs  no-remask K=16 (69%)    → can K=50 close the gap?"
        echo ""
        echo "Expected timing: ~10-13 min for 600 samples (vectorized scoring)"
        ;;
esac
