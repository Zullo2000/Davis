#!/bin/bash
# =============================================================================
# G5 K* sweep: find minimal K for good constraint satisfaction
# =============================================================================
#
# Fixed: α=0.1, soft reward, v1 loglinear checkpoint
# Sweep: K ∈ {4, 8, 10, 12, 14, 16, 20, 24}
# Seeds: [42, 123] (2 of 5), 100 samples/seed = 200 total per config
#
# Run variants sequentially to cope with GPU contention:
#   Phase 1: no-remasking (8 runs)
#   Phase 2: confidence remasking (8 runs) — after reviewing Phase 1 results
#
# Usage:
#   cd BD_Generation
#   bash scripts/run_g5_kstar.sh noremask all      # Phase 1: full pipeline
#   bash scripts/run_g5_kstar.sh confidence all     # Phase 2: full pipeline
#
#   # Or step by step:
#   bash scripts/run_g5_kstar.sh noremask calibrate
#   bash scripts/run_g5_kstar.sh noremask generate
#   bash scripts/run_g5_kstar.sh noremask evaluate
#   bash scripts/run_g5_kstar.sh noremask compare
#   bash scripts/run_g5_kstar.sh noremask analyze
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

# Reduced samples: 2 seeds × 100 samples
HYDRA_REDUCED="eval.seeds=[42,123] eval.num_samples=100"

# Sweep
ALPHA=0.1
KS=(4 8 10 12 14 16 20 24)

BASELINE_NOREMASK="llada_topp0.9_no_remask"
BASELINE_CONFIDENCE="llada_topp0.9_remdm_confidence_tsw1.0"

TAG="kstar"

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
        COMPARISON_FILE="eval_results/loglinear_noise_sc/comparison_guided_kstar_noremask.md"
        ;;
    confidence)
        HYDRA_VARIANT="$HYDRA_CONFIDENCE"
        CAL_FILE="$CAL_FILE_CONFIDENCE"
        BASELINE="$BASELINE_CONFIDENCE"
        VARIANT_LABEL="confidence remasking (tsw=1.0)"
        COMPARISON_FILE="eval_results/loglinear_noise_sc/comparison_guided_kstar_confidence.md"
        ;;
    *)
        echo "G5 K* sweep: find minimal K for good constraint satisfaction"
        echo ""
        echo "Usage: bash scripts/run_g5_kstar.sh <variant> <step>"
        echo ""
        echo "Variants:"
        echo "  noremask     No-remasking variant (Phase 1 — run first)"
        echo "  confidence   Confidence remasking tsw=1.0 (Phase 2 — run after reviewing Phase 1)"
        echo ""
        echo "Steps:"
        echo "  calibrate   Calibrate P90 normalizers (CPU, ~30s)"
        echo "  generate    Generate 8 guided configs (GPU, ~45min-1h under contention)"
        echo "  evaluate    Evaluate baseline + 8 guided (CPU)"
        echo "  compare     Generate comparison table (CPU)"
        echo "  analyze     Outlier-aware analysis plots (CPU)"
        echo "  all         Run all steps sequentially"
        echo ""
        echo "Fixed: α=$ALPHA, soft reward, v1 loglinear"
        echo "Sweep: K ∈ {${KS[*]}}"
        echo "Seeds: [42, 123], 100 samples/seed = 200 per config"
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
    print_header "Generate 8 guided configs (GPU)"
    echo "  K sweep: {${KS[*]}}"
    echo "  α = $ALPHA (fixed)"
    echo "  Seeds: [42, 123], 100 samples/seed"

    local run=1
    local total=${#KS[@]}

    for k in "${KS[@]}"; do
        echo ""
        echo "--- Run $run/$total: K=$k α=$ALPHA ---"
        python scripts/generate_guided.py \
            eval.checkpoint_path="$V1_CKPT" \
            $HYDRA_COMMON \
            $HYDRA_VARIANT \
            $HYDRA_REDUCED \
            $COMMON \
            --guidance-config "$GUIDANCE_CONFIG" \
            --calibration "$CAL_FILE" \
            --alpha "$ALPHA" --K "$k" \
            --guidance-tag "$TAG"
        run=$((run + 1))
    done

    echo ""
    echo "Generation complete: $total runs."
}

# --- Step 3: Evaluate ---
step_evaluate() {
    print_header "Evaluate baseline + 8 guided models (CPU)"

    # Re-evaluate the unguided baseline WITH constraint metrics
    echo "--- Evaluating baseline: $BASELINE ---"
    python scripts/evaluate.py \
        --schedule loglinear_noise_sc \
        --model "$BASELINE" \
        --guidance-config "$GUIDANCE_CONFIG"

    # Evaluate guided models
    for k in "${KS[@]}"; do
        local model="${BASELINE}_guided_${TAG}_K${k}_a${ALPHA}"
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
    for k in "${KS[@]}"; do
        models+=("${BASELINE}_guided_${TAG}_K${k}_a${ALPHA}")
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

    for k in "${KS[@]}"; do
        local model="${BASELINE}_guided_${TAG}_K${k}_a${ALPHA}"
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
