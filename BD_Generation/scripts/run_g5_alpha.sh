#!/bin/bash
# =============================================================================
# G5 Round 3: α fine-tuning at K=16
# =============================================================================
#
# Fixed: K=16, soft reward, v1 loglinear checkpoint
# Sweep: α ∈ {0.01, 0.05, 0.1, 0.15, 0.3, 0.5}
# Seeds: [42, 123, 456] (3 seeds), 200 samples/seed = 600 total per config
#
# Run variants sequentially:
#   Phase 1: no-remasking (6 runs)
#   Phase 2: confidence remasking (6 runs)
#
# Usage:
#   cd BD_Generation
#   bash scripts/run_g5_alpha.sh noremask all      # Phase 1: full pipeline
#   bash scripts/run_g5_alpha.sh confidence all     # Phase 2: full pipeline
#
#   # Or step by step:
#   bash scripts/run_g5_alpha.sh noremask calibrate
#   bash scripts/run_g5_alpha.sh noremask generate
#   bash scripts/run_g5_alpha.sh noremask evaluate
#   bash scripts/run_g5_alpha.sh noremask compare
#   bash scripts/run_g5_alpha.sh noremask analyze
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

# Sweep
K=16
ALPHAS=(0.01 0.05 0.1 0.15 0.3 0.5)

BASELINE_NOREMASK="llada_topp0.9_no_remask"
BASELINE_CONFIDENCE="llada_topp0.9_remdm_confidence_tsw1.0"

TAG="alpha"

# =============================================================================
# Resolve variant from first argument
# =============================================================================

VARIANT="${1:-help}"
STEP="${2:-help}"

case "$VARIANT" in
    noremask)
        HYDRA_VARIANT="$HYDRA_NOREMASK"
        CAL_FILE="$CAL_FILE_NOREMASK"
        BASELINE="$BASELINE_NOREMASK"
        VARIANT_LABEL="no-remasking"
        COMPARISON_FILE="eval_results/loglinear_noise_sc/comparison_guided_alpha_noremask.md"
        ;;
    confidence)
        HYDRA_VARIANT="$HYDRA_CONFIDENCE"
        CAL_FILE="$CAL_FILE_CONFIDENCE"
        BASELINE="$BASELINE_CONFIDENCE"
        VARIANT_LABEL="confidence remasking (tsw=1.0)"
        COMPARISON_FILE="eval_results/loglinear_noise_sc/comparison_guided_alpha_confidence.md"
        ;;
    *)
        echo "G5 Round 3: α fine-tuning at K=16"
        echo ""
        echo "Usage: bash scripts/run_g5_alpha.sh <variant> <step>"
        echo ""
        echo "Variants:"
        echo "  noremask     No-remasking variant (Phase 1 — run first)"
        echo "  confidence   Confidence remasking tsw=1.0 (Phase 2 — run after Phase 1)"
        echo ""
        echo "Steps:"
        echo "  calibrate   Calibrate P90 normalizers (CPU, ~30s)"
        echo "  generate    Generate 6 guided configs (GPU, ~2-3h per variant at 600 samples/config)"
        echo "  evaluate    Evaluate baseline + 6 guided (CPU)"
        echo "  compare     Generate comparison table (CPU)"
        echo "  analyze     Outlier-aware analysis plots (CPU)"
        echo "  all         Run all steps sequentially"
        echo ""
        echo "Fixed: K=$K, soft reward, v1 loglinear"
        echo "Sweep: α ∈ {${ALPHAS[*]}}"
        echo "Seeds: [42, 123, 456], 200 samples/seed = 600 per config"
        exit 0
        ;;
esac

# =============================================================================

print_header() {
    echo ""
    echo "============================================"
    echo "  $1"
    echo "  Variant: $VARIANT_LABEL"
    echo "  $(date)"
    echo "============================================"
}

# --- Step 1: Calibrate ---
step_calibrate() {
    print_header "Calibrate P90 normalizers"

    python scripts/calibrate_constraints.py \
        --schedule loglinear_noise_sc \
        --model "$BASELINE" \
        --constraints "$GUIDANCE_CONFIG" \
        --output "$CAL_FILE"

    echo ""
    echo "Calibration saved: $CAL_FILE"
    cat "$CAL_FILE"
}

# --- Step 2: Generate guided samples ---
step_generate() {
    print_header "Generate 6 guided configs (GPU)"
    echo "  α sweep: {${ALPHAS[*]}}"
    echo "  K = $K (fixed)"
    echo "  Seeds: [42, 123, 456], 200 samples/seed"

    local run=1
    local total=${#ALPHAS[@]}

    for a in "${ALPHAS[@]}"; do
        echo ""
        echo "--- Run $run/$total: α=$a K=$K ---"
        echo "    Started: $(date)"
        python scripts/generate_guided.py \
            eval.checkpoint_path="$V1_CKPT" \
            $HYDRA_COMMON \
            $HYDRA_VARIANT \
            $HYDRA_SAMPLES \
            $COMMON \
            --guidance-config "$GUIDANCE_CONFIG" \
            --calibration "$CAL_FILE" \
            --alpha "$a" --K "$K" \
            --guidance-tag "$TAG"
        echo "    Finished: $(date)"
        run=$((run + 1))
    done

    echo ""
    echo "Generation complete: $total runs."
}

# --- Step 3: Evaluate ---
step_evaluate() {
    print_header "Evaluate baseline + 6 guided models (CPU)"

    # Re-evaluate the unguided baseline WITH constraint metrics
    echo "--- Evaluating baseline: $BASELINE ---"
    python scripts/evaluate.py \
        --schedule loglinear_noise_sc \
        --model "$BASELINE" \
        --guidance-config "$GUIDANCE_CONFIG"

    # Evaluate guided models
    for a in "${ALPHAS[@]}"; do
        local model="${BASELINE}_guided_${TAG}_K${K}_a${a}"
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

# --- Step 4: Comparison table ---
step_compare() {
    print_header "Generate comparison table"

    local models=("$BASELINE")
    for a in "${ALPHAS[@]}"; do
        models+=("${BASELINE}_guided_${TAG}_K${K}_a${a}")
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

    for a in "${ALPHAS[@]}"; do
        local model="${BASELINE}_guided_${TAG}_K${K}_a${a}"
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
        echo "Unknown step: $STEP"
        echo "Valid steps: calibrate, generate, evaluate, compare, analyze, all"
        exit 1
        ;;
esac
