#!/bin/bash
# filepath: /home/egm/Data/Projects/CopGen/leave-one-out-evaluations.sh

echo "Starting evaluations..."
echo "Working directory: $(pwd)"
echo "Timestamp: $(date)"
echo ""

DATASET_ROOT="./data/majorTOM/test_dataset_copgen_leave_one_out"

# Automatically discover all unique seeds from existing run directories
SEEDS=$(ls -d "$DATASET_ROOT/outputs/terramind"/input_*_output_*_seed_* 2>/dev/null | sed 's/.*_seed_//' | sort -u -r)

if [ -z "$SEEDS" ]; then
    echo "No run directories found in $DATASET_ROOT/outputs/terramind/"
    exit 1
fi

echo "Discovered seeds: $SEEDS"
echo ""

evaluate() {
    OUTPUT=$1
    shift
    INPUTS=("$@")
    INPUT_STRING=$(printf "_%s" "${INPUTS[@]}")
    INPUT_STRING=${INPUT_STRING:1}  # remove leading underscore

    RUN_DIR="$DATASET_ROOT/outputs/terramind/input_${INPUT_STRING}_output_${OUTPUT}_seed_${SEED}"

    # Skip if metrics already exist (avoid Python startup overhead)
    if [[ -f "$RUN_DIR/output_metrics.txt" && -f "$RUN_DIR/output_metrics_per_tile.csv" ]]; then
        echo -e "\033[32mSkipping (already evaluated): $RUN_DIR\033[0m"
        return 0
    fi

    echo -e "\033[34mEvaluating: $RUN_DIR\033[0m"

    python -m benchmark.evaluation.evaluate_terramind_run \
        --dataset-dir $DATASET_ROOT \
        --run-dir "$RUN_DIR" \
        --verbose --per-tile
}

for SEED in $SEEDS; do
    echo -e "\033[31mRunning evaluations for seed $SEED\033[0m"

    # ----------------------
    # target: DEM
    # ----------------------
    evaluate DEM LULC S1RTC S2L2A S2L1C
    evaluate DEM coords S1RTC S2L2A S2L1C
    evaluate DEM coords LULC S2L2A S2L1C
    evaluate DEM coords LULC S1RTC S2L1C
    evaluate DEM coords LULC S1RTC S2L2A

    # ----------------------
    # target: LULC
    # ----------------------
    evaluate LULC DEM S1RTC S2L2A S2L1C
    evaluate LULC DEM coords S2L2A S2L1C
    evaluate LULC DEM coords S1RTC S2L1C
    evaluate LULC DEM coords S1RTC S2L2A
    evaluate LULC coords S1RTC S2L2A S2L1C

    # ----------------------
    # target: S1RTC
    # ----------------------
    evaluate S1RTC DEM LULC S2L2A S2L1C
    evaluate S1RTC DEM LULC coords S2L1C
    evaluate S1RTC DEM LULC coords S2L2A
    evaluate S1RTC coords LULC S2L2A S2L1C
    evaluate S1RTC DEM coords S2L2A S2L1C

    # ----------------------
    # target: S2L1C
    # ----------------------
    evaluate S2L1C DEM LULC S1RTC S2L2A
    evaluate S2L1C DEM LULC coords S2L2A
    evaluate S2L1C DEM LULC coords S1RTC
    evaluate S2L1C coords LULC S1RTC S2L2A
    evaluate S2L1C DEM coords S1RTC S2L2A

    # ----------------------
    # target: S2L2A
    # ----------------------
    evaluate S2L2A DEM LULC S1RTC S2L1C
    evaluate S2L2A DEM LULC coords S2L1C
    evaluate S2L2A DEM LULC coords S1RTC
    evaluate S2L2A coords LULC S1RTC S2L1C
    evaluate S2L2A DEM coords S1RTC S2L1C

    # ----------------------
    # target: coords
    # ----------------------
    evaluate coords DEM S1RTC S2L2A S2L1C
    evaluate coords LULC S1RTC S2L2A S2L1C
    evaluate coords DEM LULC S2L2A S2L1C
    evaluate coords DEM LULC S1RTC S2L2A
    evaluate coords DEM LULC S1RTC S2L1C

done

echo "All evaluations completed at $(date)"
