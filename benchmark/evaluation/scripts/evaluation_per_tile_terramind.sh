#!/bin/bash

DATASET_ROOT="./data/majorTOM/test_dataset_copgen_leave_one_out"
TERRAMIND_DIR="$DATASET_ROOT/outputs/terramind"

for folder in "$TERRAMIND_DIR"/*; do
    if [ -d "$folder" ]; then
        echo "Running evaluation for: $folder"
        python3 benchmark/evaluation/evaluate_terramind_run.py \
            --dataset-dir $DATASET_ROOT \
            --run-dir "$folder" \
            --per-tile
    fi
done


DATASET_ROOT="./data/majorTOM/test_dataset_copgen"
TERRAMIND_DIR="$DATASET_ROOT/outputs/terramind"

for folder in "$TERRAMIND_DIR"/*; do
    if [ -d "$folder" ]; then
        echo "Running evaluation for: $folder"
        python3 benchmark/evaluation/evaluate_terramind_run.py \
            --dataset-dir $DATASET_ROOT \
            --run-dir "$folder" \
            --per-tile
    fi
done