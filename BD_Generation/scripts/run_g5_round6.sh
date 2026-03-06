#!/bin/bash
# =============================================================================
# G5 Round 6: Option A + EMA Lock vs Option B + EMA Lock
# =============================================================================
#
# Purpose: Test adaptive EMA-based remasking lock with confidence remasking.
# Hypothesis: remasking helps early (exploration) but hurts later (exploitation).
# EMA lock starts with remasking ON, tracks per-sample EMA(reward), and
# permanently disables remasking when reward stops improving.
#
# Lock criterion: d(EMA_reward) <= 0 for 3 consecutive steps OR t=0.5 deadline.
# EMA beta = 0.85 (fixed, no sweep).
#
# Fixed: K=16, alpha=0.01, v1 loglinear checkpoint, soft reward only
# Grid:  2 configs:
#   1. confidence + Option A (fresh logits) + EMA lock
#   2. confidence + Option B (protect just-unmasked) + EMA lock
#
# Comparison targets from prior rounds (already generated, not re-run):
#   - no-remask + soft               = 69% (upper bound, Round 4)
#   - confidence + Option C + soft   = 56% (Round 4 baseline)
#   - confidence + Option A + soft   = 55% (Round 5, no lock)
#   - confidence + Option B + soft   = 56% (Round 5, no lock)
#
# Seeds: [42, 123, 456] (3 seeds), 200 samples/seed = 600 per config
#
# Results saved to: eval_results/loglinear_noise_sc/round6_guid/
#
# Usage:
#   cd BD_Generation
#   bash scripts/run_g5_round6.sh calibrate    # Step 1: calibrate (CPU, ~30s)
#   bash scripts/run_g5_round6.sh generate     # Step 2: generate 2 configs (GPU)
#   bash scripts/run_g5_round6.sh evaluate     # Step 3: evaluate (CPU)
#   bash scripts/run_g5_round6.sh compare      # Step 4: comparison table (CPU)
#   bash scripts/run_g5_round6.sh analyze      # Step 5: analysis plots (CPU)
#   bash scripts/run_g5_round6.sh organize     # Step 6: move results to round6_guid/
#   bash scripts/run_g5_round6.sh all          # Steps 1-6 sequentially
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

# 3 seeds x 200 samples = 600 per config
HYDRA_SAMPLES="eval.seeds=[42,123,456] eval.num_samples=200"

# Fixed hyperparameters
K=16
ALPHA=0.01

# EMA lock parameters
EMA_BETA=0.85
EMA_CONSECUTIVE=3
EMA_DEADLINE=0.5

# Baselines (from prior rounds, already generated)
BASELINE_NOREMASK="llada_topp0.9_no_remask"
BASELINE_CONFIDENCE="llada_topp0.9_remdm_confidence_tsw1.0"

# Round 4 models (for comparison)
R4_NOREMASK_SOFT="${BASELINE_NOREMASK}_guided_r4soft_K${K}_a${ALPHA}"
R4_CONFIDENCE_C_SOFT="${BASELINE_CONFIDENCE}_guided_r4soft_K${K}_a${ALPHA}"

# Round 5 models (for comparison)
R5_OPTA="${BASELINE_CONFIDENCE}_guided_r5optA_K${K}_a${ALPHA}"
R5_OPTB="${BASELINE_CONFIDENCE}_guided_r5optB_K${K}_a${ALPHA}"

# Round 6 models
R6_OPTA="${BASELINE_CONFIDENCE}_guided_r6lockA_K${K}_a${ALPHA}"
R6_OPTB="${BASELINE_CONFIDENCE}_guided_r6lockB_K${K}_a${ALPHA}"

# Output
SCHEDULE="loglinear_noise_sc"
EVAL_DIR="eval_results/${SCHEDULE}"
ROUND6_DIR="${EVAL_DIR}/round6_guid"
COMPARISON_FILE="${ROUND6_DIR}/comparison_guided_round6.md"

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

# --- Step 2: Generate 2 guided configs with EMA lock ---
step_generate() {
    print_header "Generate 2 guided configs with EMA lock (GPU)"
    echo "  K=$K, alpha=$ALPHA, soft reward only"
    echo "  EMA lock: beta=$EMA_BETA, consecutive=$EMA_CONSECUTIVE, deadline=$EMA_DEADLINE"
    echo "  Config 1: confidence + Option A (fresh logits) + EMA lock"
    echo "  Config 2: confidence + Option B (protect just-unmasked) + EMA lock"
    echo "  Seeds: [42, 123, 456], 200 samples/seed"

    # --- Config 1: confidence + Option A + EMA lock ---
    echo ""
    echo "--- Run 1/2: confidence + Option A + EMA lock ---"
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
        --ema-lock \
        --ema-beta "$EMA_BETA" \
        --ema-lock-consecutive "$EMA_CONSECUTIVE" \
        --ema-lock-deadline "$EMA_DEADLINE" \
        --guidance-tag "r6lockA"
    echo "    Finished: $(date)"

    # --- Config 2: confidence + Option B + EMA lock ---
    echo ""
    echo "--- Run 2/2: confidence + Option B + EMA lock ---"
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
        --ema-lock \
        --ema-beta "$EMA_BETA" \
        --ema-lock-consecutive "$EMA_CONSECUTIVE" \
        --ema-lock-deadline "$EMA_DEADLINE" \
        --guidance-tag "r6lockB"
    echo "    Finished: $(date)"

    echo ""
    echo "Generation complete: 2 runs."
}

# --- Step 3: Evaluate (2 new guided models) ---
step_evaluate() {
    print_header "Evaluate 2 new guided models (CPU)"

    local r6_models=("$R6_OPTA" "$R6_OPTB")

    for model in "${r6_models[@]}"; do
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
    print_header "Generate comparison table (prior rounds + Round 6)"

    # Create output directory
    mkdir -p "$ROUND6_DIR"

    # Include prior baselines + Round 5 (no lock) + Round 6 (with lock)
    local models=(
        "$BASELINE_NOREMASK"
        "$BASELINE_CONFIDENCE"
        "$R4_NOREMASK_SOFT"
        "$R4_CONFIDENCE_C_SOFT"
        "$R5_OPTA"
        "$R5_OPTB"
        "$R6_OPTA"
        "$R6_OPTB"
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

# --- Step 5: Outlier-aware analysis + EMA/lock visualization ---
step_analyze() {
    print_header "Outlier-aware analysis with EMA + lock plots"

    local r6_models=("$R6_OPTA" "$R6_OPTB")

    for model in "${r6_models[@]}"; do
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

# --- Step 6: Organize results into round6_guid/ ---
step_organize() {
    print_header "Organize Round 6 results into ${ROUND6_DIR}/"

    mkdir -p "$ROUND6_DIR"

    # Move Round 6 samples, metrics, and plots
    for model in "$R6_OPTA" "$R6_OPTB"; do
        for ext in _samples.pt .json; do
            src="${EVAL_DIR}/${model}${ext}"
            if [ -f "$src" ]; then
                mv "$src" "$ROUND6_DIR/"
                echo "  Moved: $(basename "$src")"
            fi
        done
        # Move analysis PNGs
        for png in "${EVAL_DIR}/${model}"_trajectories_*.png; do
            if [ -f "$png" ]; then
                mv "$png" "$ROUND6_DIR/"
                echo "  Moved: $(basename "$png")"
            fi
        done
        for png in "${EVAL_DIR}/${model}"_analysis_*.png; do
            if [ -f "$png" ]; then
                mv "$png" "$ROUND6_DIR/"
                echo "  Moved: $(basename "$png")"
            fi
        done
    done

    echo ""
    echo "Results organized in: $ROUND6_DIR/"
    ls -la "$ROUND6_DIR/" 2>/dev/null || true
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
        echo "G5 Round 6: Option A + EMA Lock vs Option B + EMA Lock"
        echo ""
        echo "Usage: bash scripts/run_g5_round6.sh <step>"
        echo ""
        echo "Steps:"
        echo "  calibrate   Reuse/create P90 normalizers (CPU, ~30s)"
        echo "  generate    Generate 2 guided configs with EMA lock (GPU)"
        echo "  evaluate    Evaluate 2 new guided models (CPU)"
        echo "  compare     Comparison table: prior rounds + R6 (CPU)"
        echo "  analyze     Outlier-aware analysis + EMA/lock plots (CPU)"
        echo "  organize    Move Round 6 results to ${ROUND6_DIR}/"
        echo "  all         Run all steps sequentially"
        echo ""
        echo "Grid: 2 configs (Option A + lock, Option B + lock)"
        echo "Fixed: K=$K, alpha=$ALPHA, soft reward, v1 loglinear"
        echo "EMA lock: beta=$EMA_BETA, consecutive=$EMA_CONSECUTIVE, deadline=$EMA_DEADLINE"
        echo "Seeds: [42, 123, 456], 200 samples/seed = 600 per config"
        echo ""
        echo "Compares against prior results:"
        echo "  - no-remask + soft                = 69% (Round 4)"
        echo "  - confidence + Option C + soft    = 56% (Round 4)"
        echo "  - confidence + Option A + soft    = 55% (Round 5, no lock)"
        echo "  - confidence + Option B + soft    = 56% (Round 5, no lock)"
        ;;
esac
