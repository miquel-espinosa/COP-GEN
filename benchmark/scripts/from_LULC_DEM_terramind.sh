#!/bin/bash

echo "Starting generations..."
echo "Working directory: $(pwd)"
echo "Timestamp: $(date)"
echo ""

DATASET_ROOT="./data/majorTOM/test_dataset_copgen_from_DEM_LULC"
SEED_START=0
SEED_END=15

# Modalities: DEM LULC S1RTC S2L2A S2L1C coords

for SEED in $(seq $SEED_START $SEED_END); do
    echo -e "\033[31mRunning experiment with seed $SEED\033[0m"

    python terramind_cli.py --input DEM LULC --output S2L2A  --dataset_root $DATASET_ROOT --seed $SEED
    python terramind_cli.py --input DEM LULC --output S2L1C  --dataset_root $DATASET_ROOT --seed $SEED
    python terramind_cli.py --input DEM LULC --output S1RTC  --dataset_root $DATASET_ROOT --seed $SEED
    
done

echo "Completed at $(date)"