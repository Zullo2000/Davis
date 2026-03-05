#!/bin/bash
# =============================================================================
# G5 Round 5: Option A vs Option B comparison at K=16, α=0.01
# =============================================================================
#
# Purpose: Test the remaining remasking mitigation options (A and B) for
# confidence remasking. Round 4 showed Option C (RACB) alone is insufficient
# (56% vs 69% no-remask). This round tests whether Option A (fresh logits)
# or Option B (protect just-unmasked) can rescue confidence remasking.
#
# Fixed: K=16, α=0.01, v1 loglinear checkpoint, soft reward only
# Grid:  2 configs (confidence remasking with different mitigations)
#   1. confidence + Option A (fresh logits)          -- --fresh-logits-remask
#   2. confidence + Option B (protect just-unmasked) -- --protect-just-unmasked
#
# Comparison targets from Round 4 (already generated, not re-run):
#   - no-remask + soft             = 69% (upper bound)
#   - confidence + Option C + soft = 56% (baseline for mitigation comparison)
#
# Seeds: [42, 123, 456] (3 seeds), 200 samples/seed = 600 per config
#
# Usage:
#   cd BD_Generation
#   bash scripts/run_g5_round5.sh calibrate    # Step 1: calibrate (CPU, ~30s)
#   bash scripts/run_g5_round5.sh generate     # Step 2: generate 2 configs (GPU)
#   bash scripts/run_g5_round5.sh evaluate     # Step 3: evaluate 2 new + reuse R4 (CPU)
#   bash scripts/run_g5_round5.sh compare      # Step 4: comparison table (CPU)
#   bash scripts/run_g5_round5.sh analyze      # Step 5: outlier-aware analysis (CPU)
#   bash scripts/run_g5_round5.sh all          # Steps 1-5 sequentially
# =============================================================================

set -euo pipefail

# --- Configuration ---
GUIDANCE_CONFIG="configs/guidance/example_basic.yaml"
COMMON="wandb.mode=disabled"

V1_CKPT="outputs/2026-02-19_16-58-23/checkpoints/checkpoint_final.pt"
CAL_FILE_CONFIDENCE="configs/guidance/calibration_v1_confidence.json"
CAL_FILE_NOREMASK="configs/guidance/calibration_v1_no_remask.json"

# Shared Hydra overrides
HYDRA_COMMON="noise=loglinear eval.unmasking_mode=llada eval.top_p=0.9"

# Variant-specific Hydra overrides
HYDRA_CONFIDENCE="eval.remasking.enabled=true eval.remasking.strategy=confidence eval.remasking.t_switch=1.0"

# 3 seeds × 200 samples = 600 per config
HYDRA_SAMPLES="eval.seeds=[42,123,456] eval.num_samples=200"

# Fixed hyperparameters
K=16
ALPHA=0.01

# Baselines (from Round 4, already generated)
BASELINE_NOREMASK="llada_topp0.9_no_remask"
BASELINE_CONFIDENCE="llada_topp0.9_remdm_confidence_tsw1.0"

# Round 4 models (for comparison, already generated)
R4_NOREMASK_SOFT="${BASELINE_NOREMASK}_guided_r4soft_K${K}_a${ALPHA}"
R4_CONFIDENCE_C_SOFT="${BASELINE_CONFIDENCE}_guided_r4soft_K${K}_a${ALPHA}"

# Output
COMPARISON_FILE="eval_results/loglinear_noise_sc/comparison_guided_round5.md"

# =============================================================================

print_header() {
    echo ""
    echo "============================================"
    echo "  $1"
    echo "  $(date)"
    echo "============================================"
}

# --- Step 1: Calibrate (reuse Round 4 calibration if available) ---
step_calibrate() {
    print_header "Calibrate P90 normalizers (confidence variant)"

    if [ -f "$CAL_FILE_CONFIDENCE" ]; then
        echo "Calibration file already exists: $CAL_FILE_CONFIDENCE"
        cat "$CAL_FILE_CONFIDENCE"
        echo ""
        echo "Skipping calibration (reusing Round 4 calibration)."
    else
        echo "--- Calibrating: confidence baseline ---"
        python scripts/calibrate_constraints.py \
            --schedule loglinear_noise_sc \
            --model "$BASELINE_CONFIDENCE" \
            --constraints "$GUIDANCE_CONFIG" \
            --output "$CAL_FILE_CONFIDENCE"
        echo "Saved: $CAL_FILE_CONFIDENCE"
        cat "$CAL_FILE_CONFIDENCE"
    fi
}

# --- Step 2: Generate 2 guided configs ---
step_generate() {
    print_header "Generate 2 guided configs (GPU)"
    echo "  K=$K, α=$ALPHA, soft reward only"
    echo "  Config 1: confidence + Option A (fresh logits)"
    echo "  Config 2: confidence + Option B (protect just-unmasked)"
    echo "  Seeds: [42, 123, 456], 200 samples/seed"

    # --- Config 1: confidence + Option A (fresh logits) ---
    echo ""
    echo "--- Run 1/2: confidence + Option A (fresh logits) ---"
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
        --guidance-tag "r5optA"
    echo "    Finished: $(date)"

    # --- Config 2: confidence + Option B (protect just-unmasked) ---
    echo ""
    echo "--- Run 2/2: confidence + Option B (protect just-unmasked) ---"
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
        --guidance-tag "r5optB"
    echo "    Finished: $(date)"

    echo ""
    echo "Generation complete: 2 runs."
}

# --- Step 3: Evaluate (2 new guided models) ---
step_evaluate() {
    print_header "Evaluate 2 new guided models (CPU)"

    # Round 5 models
    local r5_models=(
        "${BASELINE_CONFIDENCE}_guided_r5optA_K${K}_a${ALPHA}"
        "${BASELINE_CONFIDENCE}_guided_r5optB_K${K}_a${ALPHA}"
    )

    for model in "${r5_models[@]}"; do
        echo ""
        echo "--- Evaluating: $model ---"
        python scripts/evaluate.py \
            --schedule loglinear_noise_sc \
            --model "$model" \
            --guidance-config "$GUIDANCE_CONFIG"
    done

    echo ""
    echo "Evaluation complete."
}

# --- Step 4: Comparison table (Round 4 + Round 5) ---
step_compare() {
    print_header "Generate comparison table (Round 4 baselines + Round 5 new)"

    # Include Round 4 references + Round 5 new configs
    local models=(
        "$BASELINE_NOREMASK"
        "$BASELINE_CONFIDENCE"
        "$R4_NOREMASK_SOFT"
        "$R4_CONFIDENCE_C_SOFT"
        "${BASELINE_CONFIDENCE}_guided_r5optA_K${K}_a${ALPHA}"
        "${BASELINE_CONFIDENCE}_guided_r5optB_K${K}_a${ALPHA}"
    )

    echo "Comparing ${#models[@]} models:"
    for m in "${models[@]}"; do echo "  - $m"; done

    python scripts/compare_selected.py \
        --schedule loglinear_noise_sc \
        --models "${models[@]}" \
        --guided \
        --output "$COMPARISON_FILE"

    echo ""
    echo "Comparison table: $COMPARISON_FILE"
}

# --- Step 5: Outlier-aware analysis ---
step_analyze() {
    print_header "Outlier-aware analysis (--plot-analysis)"

    local r5_models=(
        "${BASELINE_CONFIDENCE}_guided_r5optA_K${K}_a${ALPHA}"
        "${BASELINE_CONFIDENCE}_guided_r5optB_K${K}_a${ALPHA}"
    )

    for model in "${r5_models[@]}"; do
        echo ""
        echo "--- Analyzing: $model ---"
        python scripts/analyze_guidance_stats.py \
            --schedule loglinear_noise_sc \
            --model "$model" \
            --plot-analysis
    done

    echo ""
    echo "Analysis complete. Check eval_results/loglinear_noise_sc/ for *_trajectories_*.png"
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
    all)
        step_calibrate
        step_generate
        step_evaluate
        step_compare
        step_analyze
        ;;
    *)
        echo "G5 Round 5: Option A vs Option B comparison at K=16, α=0.01"
        echo ""
        echo "Usage: bash scripts/run_g5_round5.sh <step>"
        echo ""
        echo "Steps:"
        echo "  calibrate   Reuse/create P90 normalizers (CPU, ~30s)"
        echo "  generate    Generate 2 guided configs (GPU)"
        echo "  evaluate    Evaluate 2 new guided models (CPU)"
        echo "  compare     Comparison table: R4 baselines + R5 new (CPU)"
        echo "  analyze     Outlier-aware analysis plots (CPU)"
        echo "  all         Run all steps sequentially"
        echo ""
        echo "Grid: 2 configs (confidence + Option A, confidence + Option B)"
        echo "Fixed: K=$K, α=$ALPHA, soft reward, v1 loglinear"
        echo "Seeds: [42, 123, 456], 200 samples/seed = 600 per config"
        echo ""
        echo "Compares against Round 4 results:"
        echo "  - no-remask + soft (69%)"
        echo "  - confidence + Option C + soft (56%)"
        ;;
esac
