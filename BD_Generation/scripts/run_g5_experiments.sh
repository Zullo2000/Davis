#!/bin/bash
# =============================================================================
# G5: SVDD Guided Generation — End-to-End Experiment Suite
# =============================================================================
#
# Runs the full G5 experiment design from implementation_state_T1_guidance.md:
#   Step 1: Calibrate P90 normalizers from unguided samples (CPU)
#   Step 2: Soft vs Hard reward mode comparison (GPU, 8 runs)
#   Step 3: Full α × K grid search (GPU, 60 runs)
#   Step 4: Evaluate all guided models (CPU)
#   Step 5: Generate comparison tables (CPU)
#
# Usage:
#   cd BD_Generation
#   bash scripts/run_g5_experiments.sh step1          # Calibrate only
#   bash scripts/run_g5_experiments.sh step2           # Soft vs Hard comparison
#   bash scripts/run_g5_experiments.sh step3 soft      # Full grid (soft mode)
#   bash scripts/run_g5_experiments.sh step3 hard      # Full grid (hard mode)
#   bash scripts/run_g5_experiments.sh step4           # Evaluate all
#   bash scripts/run_g5_experiments.sh step5           # Comparison tables
#   bash scripts/run_g5_experiments.sh all soft         # Steps 1-5 sequentially
#
# Review after step 2 to pick reward mode before running step 3.
#
# Estimated time:
#   Step 1: ~2 min (CPU)
#   Step 2: ~40 min (GPU, 8 × ~5 min each)
#   Step 3: ~5 hours (GPU, 60 × ~5 min each)
#   Step 4: ~10 min (CPU)
#   Step 5: ~1 min (CPU)
# =============================================================================

set -euo pipefail

# --- Configuration ---
GUIDANCE_CONFIG="configs/guidance/example_basic.yaml"
COMMON="wandb.mode=disabled"

# Checkpoints
V1_CKPT="outputs/2026-02-19_16-58-23/checkpoints/checkpoint_final.pt"
V2_CKPT="outputs/v2_2026-02-20_18-36-23/checkpoints/checkpoint_final.pt"

# Calibration output paths
CAL_V1_NR="configs/guidance/calibration_v1_no_remask.json"
CAL_V1_CR="configs/guidance/calibration_v1_confidence.json"
CAL_V2_NR="configs/guidance/calibration_v2_no_remask.json"
CAL_V2_CR="configs/guidance/calibration_v2_confidence.json"

# Hyperparameter grids
ALPHAS=(0.1 0.5 1.0 2.0 5.0)
KS=(4 8 16)

# --- Variant definitions ---
# Each variant: NAME SCHEDULE CKPT HYDRA_OVERRIDES CALIBRATION_FILE
# Using parallel arrays since bash doesn't have structs
VARIANT_NAMES=("v1_no_remask" "v1_confidence" "v2_no_remask" "v2_confidence")
VARIANT_SCHEDULES=("loglinear_noise_sc" "loglinear_noise_sc" "learned_noise_sc" "learned_noise_sc")
VARIANT_CKPTS=("$V1_CKPT" "$V1_CKPT" "$V2_CKPT" "$V2_CKPT")
VARIANT_OVERRIDES=(
    "noise=loglinear eval.unmasking_mode=llada eval.top_p=0.9 eval.remasking.enabled=false"
    "noise=loglinear eval.unmasking_mode=llada eval.top_p=0.9 eval.remasking.enabled=true eval.remasking.strategy=confidence eval.remasking.eta=0.0 eval.remasking.t_switch=1.0"
    "noise=learned eval.unmasking_mode=llada eval.top_p=0.9 eval.remasking.enabled=false"
    "noise=learned eval.unmasking_mode=llada eval.top_p=0.9 eval.remasking.enabled=true eval.remasking.strategy=confidence eval.remasking.eta=0.0 eval.remasking.t_switch=1.0"
)
VARIANT_CALS=("$CAL_V1_NR" "$CAL_V1_CR" "$CAL_V2_NR" "$CAL_V2_CR")

# Calibration source models (existing unguided samples)
CAL_SCHEDULES=("loglinear_noise_sc" "loglinear_noise_sc" "learned_noise_sc" "learned_noise_sc")
CAL_MODELS=("llada_topp0.9_no_remask" "llada_topp0.9_remdm_confidence_tsw1.0" "v2_llada_topp0.9_no_remask" "v2_llada_topp0.9_remdm_confidence_tsw1.0")


# =============================================================================
# Helper functions
# =============================================================================

print_header() {
    echo ""
    echo "============================================"
    echo "  $1"
    echo "  $(date)"
    echo "============================================"
}

print_step() {
    echo ""
    echo "--- $1 ---"
}

run_guided() {
    # Args: variant_idx alpha K reward_mode tag
    local idx=$1
    local alpha=$2
    local k=$3
    local mode=$4
    local tag=$5

    local name="${VARIANT_NAMES[$idx]}"
    local ckpt="${VARIANT_CKPTS[$idx]}"
    local overrides="${VARIANT_OVERRIDES[$idx]}"
    local cal="${VARIANT_CALS[$idx]}"

    echo "  [$name] α=$alpha K=$k mode=$mode"

    python scripts/generate_guided.py \
        eval.checkpoint_path="$ckpt" \
        $overrides \
        $COMMON \
        --guidance-config "$GUIDANCE_CONFIG" \
        --alpha "$alpha" --K "$k" \
        --reward-mode "$mode" \
        --calibration "$cal" \
        --guidance-tag "$tag"
}


# =============================================================================
# Step 1: Calibrate P90 normalizers (CPU)
# =============================================================================

step1_calibrate() {
    print_header "Step 1: Calibrate P90 normalizers (CPU)"
    echo "  Using existing unguided samples to compute P90 violation normalizers."

    for i in "${!VARIANT_NAMES[@]}"; do
        print_step "Calibrating ${VARIANT_NAMES[$i]}"
        python scripts/calibrate_constraints.py \
            --schedule "${CAL_SCHEDULES[$i]}" \
            --model "${CAL_MODELS[$i]}" \
            --constraints "$GUIDANCE_CONFIG" \
            --output "${VARIANT_CALS[$i]}"
        echo "  → Saved: ${VARIANT_CALS[$i]}"
    done

    echo ""
    echo "Step 1 complete. Calibration files:"
    for cal in "${VARIANT_CALS[@]}"; do
        if [ -f "$cal" ]; then
            echo "  ✓ $cal"
            cat "$cal"
        else
            echo "  ✗ $cal (MISSING)"
        fi
    done
}


# =============================================================================
# Step 2: Soft vs Hard comparison (GPU, 8 runs)
# =============================================================================

step2_soft_vs_hard() {
    print_header "Step 2: Soft vs Hard comparison (α=1.0, K=8)"
    echo "  Running each variant with soft and hard reward modes."
    echo "  Total: 4 variants × 2 modes = 8 runs"

    local run=1
    for i in "${!VARIANT_NAMES[@]}"; do
        for mode in soft hard; do
            print_step "Run $run/8: ${VARIANT_NAMES[$i]} [$mode]"
            run_guided "$i" 1.0 8 "$mode" "${mode}_test"
            run=$((run + 1))
        done
    done

    echo ""
    echo "============================================"
    echo "  Step 2 complete: 8 soft/hard comparison runs done."
    echo "============================================"
    echo ""
    echo "Next: evaluate with constraint metrics and compare."
    echo "  For each schedule, run:"
    echo "    python scripts/evaluate.py --schedule loglinear_noise_sc --guidance-config $GUIDANCE_CONFIG"
    echo "    python scripts/evaluate.py --schedule learned_noise_sc --guidance-config $GUIDANCE_CONFIG"
    echo ""
    echo "Then analyze ESS, reward variance, and satisfaction rates to pick reward mode."
    echo "Run step 3 with the chosen mode: bash scripts/run_g5_experiments.sh step3 <soft|hard>"
}


# =============================================================================
# Step 3: Full α × K grid (GPU, 60 runs)
# =============================================================================

step3_full_grid() {
    local mode="${1:-soft}"
    print_header "Step 3: Full grid search (mode=$mode)"
    echo "  4 variants × ${#ALPHAS[@]} α × ${#KS[@]} K = $((4 * ${#ALPHAS[@]} * ${#KS[@]})) runs"

    local run=1
    local total=$((4 * ${#ALPHAS[@]} * ${#KS[@]}))

    for i in "${!VARIANT_NAMES[@]}"; do
        for alpha in "${ALPHAS[@]}"; do
            for k in "${KS[@]}"; do
                print_step "Run $run/$total: ${VARIANT_NAMES[$i]} α=$alpha K=$k"
                run_guided "$i" "$alpha" "$k" "$mode" "basic"
                run=$((run + 1))
            done
        done
    done

    echo ""
    echo "============================================"
    echo "  Step 3 complete: $total guided generation runs done."
    echo "  Reward mode: $mode"
    echo "============================================"
}


# =============================================================================
# Step 4: Evaluate all guided models (CPU)
# =============================================================================

step4_evaluate() {
    print_header "Step 4: Evaluate all guided models (CPU)"

    for schedule in loglinear_noise_sc learned_noise_sc; do
        print_step "Evaluating $schedule"
        python scripts/evaluate.py --schedule "$schedule" \
            --guidance-config "$GUIDANCE_CONFIG" \
            --update-comparison
    done

    echo ""
    echo "Step 4 complete. Results in:"
    echo "  eval_results/loglinear_noise_sc/*.json"
    echo "  eval_results/learned_noise_sc/*.json"
}


# =============================================================================
# Step 5: Comparison tables (CPU)
# =============================================================================

step5_compare() {
    print_header "Step 5: Generate comparison tables"

    for schedule in loglinear_noise_sc learned_noise_sc; do
        echo ""
        echo "=== $schedule ==="

        # List all guided models
        local guided_models=()
        for f in eval_results/"$schedule"/*_guided_*.json; do
            [ -f "$f" ] && guided_models+=("$(basename "$f" .json)")
        done

        if [ ${#guided_models[@]} -eq 0 ]; then
            echo "  No guided models found for $schedule."
            continue
        fi

        echo "  Found ${#guided_models[@]} guided models."

        # Get unguided baselines for comparison
        local baselines=()
        if [ "$schedule" = "loglinear_noise_sc" ]; then
            baselines=("llada_topp0.9_no_remask" "llada_topp0.9_remdm_confidence_tsw1.0")
        else
            baselines=("v2_llada_topp0.9_no_remask" "v2_llada_topp0.9_remdm_confidence_tsw1.0")
        fi

        # Compare each baseline against its guided variants
        for baseline in "${baselines[@]}"; do
            local matching=()
            for gm in "${guided_models[@]}"; do
                # Match guided models that share the same base configuration
                if [[ "$gm" == *"no_remask_guided"* ]] && [[ "$baseline" == *"no_remask"* ]]; then
                    matching+=("$gm")
                elif [[ "$gm" == *"confidence_tsw1.0_guided"* ]] && [[ "$baseline" == *"confidence"* ]]; then
                    matching+=("$gm")
                fi
            done

            if [ ${#matching[@]} -gt 0 ]; then
                echo ""
                echo "  Baseline: $baseline vs ${#matching[@]} guided variants"
                python scripts/compare_selected.py \
                    --schedule "$schedule" \
                    --models "$baseline" "${matching[@]}" \
                    --output "eval_results/${schedule}/comparison_guided_${baseline}.md"
                echo "  → eval_results/${schedule}/comparison_guided_${baseline}.md"
            fi
        done
    done

    echo ""
    echo "Step 5 complete."
}


# =============================================================================
# Main dispatcher
# =============================================================================

STEP="${1:-help}"
MODE_ARG="${2:-soft}"

case "$STEP" in
    step1)
        step1_calibrate
        ;;
    step2)
        step1_calibrate  # ensure calibration exists
        step2_soft_vs_hard
        ;;
    step3)
        step3_full_grid "$MODE_ARG"
        ;;
    step4)
        step4_evaluate
        ;;
    step5)
        step5_compare
        ;;
    all)
        step1_calibrate
        step2_soft_vs_hard
        echo ""
        echo ">>> STOP: Review soft vs hard results before continuing. <<<"
        echo ">>> Re-run with: bash scripts/run_g5_experiments.sh step3 <soft|hard> <<<"
        ;;
    *)
        echo "G5 Guided Generation Experiment Suite"
        echo ""
        echo "Usage: bash scripts/run_g5_experiments.sh <step> [mode]"
        echo ""
        echo "Steps:"
        echo "  step1       Calibrate P90 normalizers (CPU, ~2 min)"
        echo "  step2       Soft vs Hard comparison (GPU, ~40 min)"
        echo "  step3 MODE  Full α×K grid (GPU, ~5 hours). MODE=soft|hard"
        echo "  step4       Evaluate all guided models (CPU, ~10 min)"
        echo "  step5       Generate comparison tables (CPU, ~1 min)"
        echo "  all         Run steps 1-2, then pause for review"
        echo ""
        echo "Typical workflow:"
        echo "  1. bash scripts/run_g5_experiments.sh step1"
        echo "  2. bash scripts/run_g5_experiments.sh step2"
        echo "  3. Review ESS, reward variance, satisfaction → pick mode"
        echo "  4. bash scripts/run_g5_experiments.sh step3 <soft|hard>"
        echo "  5. bash scripts/run_g5_experiments.sh step4"
        echo "  6. bash scripts/run_g5_experiments.sh step5"
        ;;
esac
