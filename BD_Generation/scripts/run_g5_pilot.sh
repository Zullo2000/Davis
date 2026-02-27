#!/bin/bash
# =============================================================================
# G5 Pilot: 6-config guided generation (v1 + llada + top-p=0.9 + no remasking)
# =============================================================================
#
# Runs a focused 6-config experiment to validate guidance metrics before
# committing to the full 60-config grid.
#
# Grid: α ∈ {0.1, 1.0, 5.0} × K ∈ {4, 16} = 6 runs
# Variant: v1 loglinear, llada, top-p=0.9, no remasking
# Reward mode: soft (default from YAML)
#
# Usage:
#   cd BD_Generation
#   bash scripts/run_g5_pilot.sh calibrate    # Step 1: calibrate (CPU, ~30s)
#   bash scripts/run_g5_pilot.sh generate     # Step 2: generate 6 configs (GPU, ~30 min)
#   bash scripts/run_g5_pilot.sh evaluate     # Step 3: evaluate all 7 models (CPU, ~5 min)
#   bash scripts/run_g5_pilot.sh compare      # Step 4: comparison table (CPU)
#   bash scripts/run_g5_pilot.sh all          # Steps 1-4 sequentially
# =============================================================================

set -euo pipefail

# --- Configuration ---
GUIDANCE_CONFIG="configs/guidance/example_basic.yaml"
COMMON="wandb.mode=disabled"

V1_CKPT="outputs/2026-02-19_16-58-23/checkpoints/checkpoint_final.pt"
CAL_FILE="configs/guidance/calibration_v1_no_remask.json"

HYDRA_OVERRIDES="noise=loglinear eval.unmasking_mode=llada eval.top_p=0.9 eval.remasking.enabled=false"

# Grids
ALPHAS=(0.1 1.0 5.0)
KS=(4 16)

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
    print_header "Calibrate P90 normalizers (v1 no-remask)"

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
    print_header "Generate 6 guided configs (GPU)"
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

# --- Step 3: Evaluate (baseline + 6 guided) ---
step_evaluate() {
    print_header "Evaluate baseline + 6 guided models (CPU)"

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

    # Build list of all 7 models (baseline + 6 guided)
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
        --output "eval_results/loglinear_noise_sc/comparison_guided_pilot.md"

    echo ""
    echo "Comparison table: eval_results/loglinear_noise_sc/comparison_guided_pilot.md"
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
    all)
        step_calibrate
        step_generate
        step_evaluate
        step_compare
        ;;
    *)
        echo "G5 Pilot: 6-config guided generation experiment"
        echo ""
        echo "Usage: bash scripts/run_g5_pilot.sh <step>"
        echo ""
        echo "Steps:"
        echo "  calibrate   Calibrate P90 normalizers (CPU, ~30s)"
        echo "  generate    Generate 6 guided configs (GPU, ~30 min)"
        echo "  evaluate    Evaluate baseline + 6 guided (CPU, ~5 min)"
        echo "  compare     Generate comparison table (CPU)"
        echo "  all         Run all steps sequentially"
        echo ""
        echo "Grid: α ∈ {0.1, 1.0, 5.0} × K ∈ {4, 16} = 6 configs"
        echo "Variant: v1 + llada + top-p=0.9 + no remasking"
        ;;
esac
