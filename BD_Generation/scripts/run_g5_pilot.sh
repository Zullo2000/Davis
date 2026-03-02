#!/bin/bash
# =============================================================================
# G5 Round 2: Fine α sweep with revised constraints
# =============================================================================
#
# Round 2 experiment: finer α grid + K=24, using the revised 4-constraint set
# (one_kitchen, kitchen_near_living, no_bath_kitchen, between_2_and_3_bathrooms).
#
# Grid: α ∈ {0.01, 0.05, 0.15, 0.3} × K ∈ {16, 24} = 8 runs
# Variant: v1 loglinear, llada, top-p=0.9, no remasking
# Reward mode: soft
#
# Usage:
#   cd BD_Generation
#   bash scripts/run_g5_pilot.sh calibrate    # Step 1: calibrate (CPU, ~30s)
#   bash scripts/run_g5_pilot.sh generate     # Step 2: generate 8 configs (GPU)
#   bash scripts/run_g5_pilot.sh evaluate     # Step 3: evaluate all 9 models (CPU)
#   bash scripts/run_g5_pilot.sh compare      # Step 4: comparison table (CPU)
#   bash scripts/run_g5_pilot.sh analyze      # Step 5: outlier-aware analysis (CPU)
#   bash scripts/run_g5_pilot.sh all          # Steps 1-5 sequentially
# =============================================================================

set -euo pipefail

# --- Configuration ---
GUIDANCE_CONFIG="configs/guidance/example_basic.yaml"
COMMON="wandb.mode=disabled"

V1_CKPT="outputs/2026-02-19_16-58-23/checkpoints/checkpoint_final.pt"
CAL_FILE="configs/guidance/calibration_v1_no_remask.json"

HYDRA_OVERRIDES="noise=loglinear eval.unmasking_mode=llada eval.top_p=0.9 eval.remasking.enabled=false"

# Grids
ALPHAS=(0.01 0.05 0.15 0.3)
KS=(16 24)

BASELINE_MODEL="llada_topp0.9_no_remask"

# =============================================================================

print_header() {
    echo ""
    echo "============================================"
    echo "  $1"
    echo "  $(date)"
    echo "============================================"
}

# --- Step 1: Calibrate ---
step_calibrate() {
    print_header "Calibrate P90 normalizers (v1 no-remask, revised constraints)"

    python scripts/calibrate_constraints.py \
        --schedule loglinear_noise_sc \
        --model "$BASELINE_MODEL" \
        --constraints "$GUIDANCE_CONFIG" \
        --output "$CAL_FILE"

    echo ""
    echo "Calibration saved: $CAL_FILE"
    cat "$CAL_FILE"
}

# --- Step 2: Generate guided samples ---
step_generate() {
    print_header "Generate 8 guided configs (GPU)"
    echo "  Grid: α ∈ {${ALPHAS[*]}} × K ∈ {${KS[*]}}"

    local run=1
    local total=$(( ${#ALPHAS[@]} * ${#KS[@]} ))

    for alpha in "${ALPHAS[@]}"; do
        for k in "${KS[@]}"; do
            echo ""
            echo "--- Run $run/$total: α=$alpha K=$k ---"
            python scripts/generate_guided.py \
                eval.checkpoint_path="$V1_CKPT" \
                $HYDRA_OVERRIDES \
                $COMMON \
                --guidance-config "$GUIDANCE_CONFIG" \
                --calibration "$CAL_FILE" \
                --alpha "$alpha" --K "$k" \
                --guidance-tag basic
            run=$((run + 1))
        done
    done

    echo ""
    echo "Generation complete: $total runs."
}

# --- Step 3: Evaluate (baseline + 12 guided) ---
step_evaluate() {
    print_header "Evaluate baseline + 8 guided models (CPU)"

    # Re-evaluate the unguided baseline WITH constraint metrics
    echo "--- Evaluating baseline: $BASELINE_MODEL ---"
    python scripts/evaluate.py \
        --schedule loglinear_noise_sc \
        --model "$BASELINE_MODEL" \
        --guidance-config "$GUIDANCE_CONFIG"

    # Evaluate each guided model
    for alpha in "${ALPHAS[@]}"; do
        for k in "${KS[@]}"; do
            local model="${BASELINE_MODEL}_guided_basic_K${k}_a${alpha}"
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
    print_header "Generate comparison table"

    # Build list of all models (baseline + 12 guided)
    local models=("$BASELINE_MODEL")
    for alpha in "${ALPHAS[@]}"; do
        for k in "${KS[@]}"; do
            models+=("${BASELINE_MODEL}_guided_basic_K${k}_a${alpha}")
        done
    done

    echo "Comparing ${#models[@]} models:"
    for m in "${models[@]}"; do
        echo "  - $m"
    done

    python scripts/compare_selected.py \
        --schedule loglinear_noise_sc \
        --models "${models[@]}" \
        --guided \
        --output "eval_results/loglinear_noise_sc/comparison_guided_round2.md"

    echo ""
    echo "Comparison table: eval_results/loglinear_noise_sc/comparison_guided_round2.md"
}

# --- Step 5: Outlier-aware analysis ---
step_analyze() {
    print_header "Outlier-aware analysis (--plot-analysis)"

    for alpha in "${ALPHAS[@]}"; do
        for k in "${KS[@]}"; do
            local model="${BASELINE_MODEL}_guided_basic_K${k}_a${alpha}"
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
        echo "G5 Round 2: Fine α sweep with revised constraints"
        echo ""
        echo "Usage: bash scripts/run_g5_pilot.sh <step>"
        echo ""
        echo "Steps:"
        echo "  calibrate   Calibrate P90 normalizers (CPU, ~30s)"
        echo "  generate    Generate 8 guided configs (GPU)"
        echo "  evaluate    Evaluate baseline + 8 guided (CPU)"
        echo "  compare     Generate comparison table (CPU)"
        echo "  analyze     Outlier-aware analysis plots (CPU)"
        echo "  all         Run all steps sequentially"
        echo ""
        echo "Grid: α ∈ {0.01, 0.05, 0.15, 0.3} × K ∈ {16, 24} = 8 configs"
        echo "Variant: v1 + llada + top-p=0.9 + no remasking, soft reward"
        ;;
esac
