#!/bin/bash

DATASET_ROOT="./data/majorTOM/test_dataset_copgen_leave_one_out"
COPGEN_DIR="$DATASET_ROOT/outputs/copgen"

for folder in "$COPGEN_DIR"/*; do
    if [ -d "$folder" ]; then
        echo "Running evaluation for: $folder"
        python3 benchmark/evaluation/evaluate_copgen_run.py \
            --dataset-root $DATASET_ROOT \
            --run-root "$folder" \
            --per-tile
    fi
done


DATASET_ROOT="./data/majorTOM/test_dataset_copgen"
COPGEN_DIR="$DATASET_ROOT/outputs/copgen"

# Create array of folders and iterate in reverse order
# for folder in "$COPGEN_DIR"/*; do
#     if [ -d "$folder" ]; then
#         echo "Running evaluation for: $folder"
#         python3 benchmark/evaluation/evaluate_copgen_run.py \
#             --dataset-root $DATASET_ROOT \
#             --run-root "$folder" \
#             --per-tile
#     fi
# done
folders=("$COPGEN_DIR"/*)
for ((i=${#folders[@]}-1; i>=0; i--)); do
    folder="${folders[$i]}"
    if [ -d "$folder" ]; then
        echo "Running evaluation for: $folder"
        python3 benchmark/evaluation/evaluate_copgen_run.py \
            --dataset-root $DATASET_ROOT \
            --run-root "$folder" \
            --per-tile
    fi
done