#!/bin/bash
# =============================================================================
# G5 Round 4: Remasking × Reward-mode comparison at K=16, α=0.01
# =============================================================================
#
# Purpose: Validate new remasking implementation by comparing no-remasking vs
# confidence remasking, with both soft (softmax) and hard (argmax) reward modes.
#
# Fixed: K=16, α=0.01, v1 loglinear checkpoint
# Grid:  2 variants × 2 reward modes = 4 configs
#   1. no-remasking  + soft reward
#   2. no-remasking  + hard reward
#   3. confidence    + soft reward  + Reward-Attributed Confidence Boosting
#   4. confidence    + hard reward  + Reward-Attributed Confidence Boosting
#
# Note: confidence remasking always uses Reward-Attributed Confidence Boosting
# because vanilla confidence remasking uses stale model logits that don't
# account for the guidance reweighting, fighting the guidance signal.
#
# Seeds: [42, 123, 456] (3 seeds), 200 samples/seed = 600 per config
#
# Usage:
#   cd BD_Generation
#   bash scripts/run_g5_round4.sh calibrate    # Step 1: calibrate (CPU, ~30s)
#   bash scripts/run_g5_round4.sh generate     # Step 2: generate 4 configs (GPU)
#   bash scripts/run_g5_round4.sh evaluate     # Step 3: evaluate 2 baselines + 4 guided (CPU)
#   bash scripts/run_g5_round4.sh compare      # Step 4: comparison table (CPU)
#   bash scripts/run_g5_round4.sh analyze      # Step 5: outlier-aware analysis (CPU)
#   bash scripts/run_g5_round4.sh all          # Steps 1-5 sequentially
# =============================================================================

set -euo pipefail

# --- Configuration ---
GUIDANCE_CONFIG="configs/guidance/example_basic.yaml"
COMMON="wandb.mode=disabled"

V1_CKPT="outputs/2026-02-19_16-58-23/checkpoints/checkpoint_final.pt"
CAL_FILE_NOREMASK="configs/guidance/calibration_v1_no_remask.json"
CAL_FILE_CONFIDENCE="configs/guidance/calibration_v1_confidence.json"

# Shared Hydra overrides
HYDRA_COMMON="noise=loglinear eval.unmasking_mode=llada eval.top_p=0.9"

# Variant-specific Hydra overrides
HYDRA_NOREMASK="eval.remasking.enabled=false"
HYDRA_CONFIDENCE="eval.remasking.enabled=true eval.remasking.strategy=confidence eval.remasking.t_switch=1.0"

# 3 seeds × 200 samples = 600 per config
HYDRA_SAMPLES="eval.seeds=[42,123,456] eval.num_samples=200"

# Fixed hyperparameters
K=16
ALPHA=0.01

# Reward modes
REWARD_MODES=(soft hard)

# Baselines
BASELINE_NOREMASK="llada_topp0.9_no_remask"
BASELINE_CONFIDENCE="llada_topp0.9_remdm_confidence_tsw1.0"

# Output
COMPARISON_FILE="eval_results/loglinear_noise_sc/comparison_guided_round4.md"

# =============================================================================

print_header() {
    echo ""
    echo "============================================"
    echo "  $1"
    echo "  $(date)"
    echo "============================================"
}

# --- Step 1: Calibrate (2 variants) ---
step_calibrate() {
    print_header "Calibrate P90 normalizers (both variants)"

    echo "--- Calibrating: no-remasking baseline ---"
    python scripts/calibrate_constraints.py \
        --schedule loglinear_noise_sc \
        --model "$BASELINE_NOREMASK" \
        --constraints "$GUIDANCE_CONFIG" \
        --output "$CAL_FILE_NOREMASK"
    echo "Saved: $CAL_FILE_NOREMASK"
    cat "$CAL_FILE_NOREMASK"

    echo ""
    echo "--- Calibrating: confidence baseline ---"
    python scripts/calibrate_constraints.py \
        --schedule loglinear_noise_sc \
        --model "$BASELINE_CONFIDENCE" \
        --constraints "$GUIDANCE_CONFIG" \
        --output "$CAL_FILE_CONFIDENCE"
    echo "Saved: $CAL_FILE_CONFIDENCE"
    cat "$CAL_FILE_CONFIDENCE"
}

# --- Step 2: Generate 4 guided configs ---
step_generate() {
    print_header "Generate 4 guided configs (GPU)"
    echo "  K=$K, α=$ALPHA"
    echo "  Grid: {no-remask, confidence} × {soft, hard}"
    echo "  Seeds: [42, 123, 456], 200 samples/seed"

    local run=1
    local total=4

    # --- no-remasking × {soft, hard} ---
    for rmode in "${REWARD_MODES[@]}"; do
        local tag="r4${rmode}"
        echo ""
        echo "--- Run $run/$total: no-remasking + $rmode reward ---"
        echo "    Started: $(date)"
        python scripts/generate_guided.py \
            eval.checkpoint_path="$V1_CKPT" \
            $HYDRA_COMMON \
            $HYDRA_NOREMASK \
            $HYDRA_SAMPLES \
            $COMMON \
            --guidance-config "$GUIDANCE_CONFIG" \
            --calibration "$CAL_FILE_NOREMASK" \
            --alpha "$ALPHA" --K "$K" \
            --reward-mode "$rmode" \
            --guidance-tag "$tag"
        echo "    Finished: $(date)"
        run=$((run + 1))
    done

    # --- confidence × {soft, hard} (with Reward-Attributed Confidence Boosting) ---
    for rmode in "${REWARD_MODES[@]}"; do
        local tag="r4${rmode}"
        echo ""
        echo "--- Run $run/$total: confidence + $rmode reward + RACB ---"
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
            --reward-mode "$rmode" \
            --attribution-boost \
            --guidance-tag "$tag"
        echo "    Finished: $(date)"
        run=$((run + 1))
    done

    echo ""
    echo "Generation complete: $total runs."
}

# --- Step 3: Evaluate (2 baselines + 4 guided) ---
step_evaluate() {
    print_header "Evaluate 2 baselines + 4 guided models (CPU)"

    # Evaluate unguided baselines WITH constraint metrics
    echo "--- Evaluating baseline: $BASELINE_NOREMASK ---"
    python scripts/evaluate.py \
        --schedule loglinear_noise_sc \
        --model "$BASELINE_NOREMASK" \
        --guidance-config "$GUIDANCE_CONFIG"

    echo ""
    echo "--- Evaluating baseline: $BASELINE_CONFIDENCE ---"
    python scripts/evaluate.py \
        --schedule loglinear_noise_sc \
        --model "$BASELINE_CONFIDENCE" \
        --guidance-config "$GUIDANCE_CONFIG"

    # Evaluate 4 guided models
    for baseline in "$BASELINE_NOREMASK" "$BASELINE_CONFIDENCE"; do
        for rmode in "${REWARD_MODES[@]}"; do
            local tag="r4${rmode}"
            local model="${baseline}_guided_${tag}_K${K}_a${ALPHA}"
            echo ""
            echo "--- Evaluating: $model ---"
            python scripts/evaluate.py \
                --schedule loglinear_noise_sc \
                --model "$model" \
                --guidance-config "$GUIDANCE_CONFIG"
        done
    done

    echo ""
    echo "Evaluation complete."
}

# --- Step 4: Comparison table ---
step_compare() {
    print_header "Generate comparison table (all 6 models)"

    # Build list: 2 baselines + 4 guided
    local models=(
        "$BASELINE_NOREMASK"
        "$BASELINE_CONFIDENCE"
    )
    for baseline in "$BASELINE_NOREMASK" "$BASELINE_CONFIDENCE"; do
        for rmode in "${REWARD_MODES[@]}"; do
            local tag="r4${rmode}"
            models+=("${baseline}_guided_${tag}_K${K}_a${ALPHA}")
        done
    done

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

    for baseline in "$BASELINE_NOREMASK" "$BASELINE_CONFIDENCE"; do
        for rmode in "${REWARD_MODES[@]}"; do
            local tag="r4${rmode}"
            local model="${baseline}_guided_${tag}_K${K}_a${ALPHA}"
            echo ""
            echo "--- Analyzing: $model ---"
            python scripts/analyze_guidance_stats.py \
                --schedule loglinear_noise_sc \
                --model "$model" \
                --plot-analysis
        done
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
        echo "G5 Round 4: Remasking × Reward-mode comparison at K=16, α=0.01"
        echo ""
        echo "Usage: bash scripts/run_g5_round4.sh <step>"
        echo ""
        echo "Steps:"
        echo "  calibrate   Calibrate P90 normalizers for both variants (CPU, ~30s)"
        echo "  generate    Generate 4 guided configs (GPU, ~2-3h at 600 samples/config)"
        echo "  evaluate    Evaluate 2 baselines + 4 guided (CPU)"
        echo "  compare     Generate comparison table (CPU)"
        echo "  analyze     Outlier-aware analysis plots (CPU)"
        echo "  all         Run all steps sequentially"
        echo ""
        echo "Grid: {no-remask, confidence} × {soft, hard} = 4 configs"
        echo "Fixed: K=$K, α=$ALPHA, v1 loglinear"
        echo "Seeds: [42, 123, 456], 200 samples/seed = 600 per config"
        ;;
esac
