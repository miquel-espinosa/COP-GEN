#!/bin/bash

echo "Starting generations..."
echo "Working directory: $(pwd)"
echo "Timestamp: $(date)"
echo ""

DATASET_ROOT="./data/majorTOM/test_dataset_copgen"
SEED_START=6
SEED_END=19

# Modalities: DEM LULC S1RTC S2L2A S2L1C coords

for SEED in $(seq $SEED_START $SEED_END); do
    echo -e "\033[31mRunning experiment with seed $SEED\033[0m"

    # target: coords
    python terramind_cli.py --input DEM LULC S1RTC S2L2A S2L1C    --output coords --dataset_root $DATASET_ROOT --seed $SEED

    # target: DEM
    python terramind_cli.py --input coords LULC S1RTC S2L2A S2L1C --output DEM    --dataset_root $DATASET_ROOT --seed $SEED

    # target: LULC
    python terramind_cli.py --input coords DEM S1RTC S2L2A S2L1C  --output LULC   --dataset_root $DATASET_ROOT --seed $SEED

    # target: S1RTC
    python terramind_cli.py --input coords DEM LULC S2L2A S2L1C   --output S1RTC  --dataset_root $DATASET_ROOT --seed $SEED

    # target: S2L1C
    python terramind_cli.py --input coords DEM LULC S1RTC S2L2A   --output S2L1C  --dataset_root $DATASET_ROOT --seed $SEED

    # target: S2L2A
    python terramind_cli.py --input coords DEM LULC S1RTC S2L1C   --output S2L2A  --dataset_root $DATASET_ROOT --seed $SEED

    # target: S2L1C (without S2L2A)
    python terramind_cli.py --input coords DEM LULC S1RTC         --output S2L1C  --dataset_root $DATASET_ROOT --seed $SEED

    # target: S2L2A (without S2L1C)
    python terramind_cli.py --input coords DEM LULC S1RTC         --output S2L2A  --dataset_root $DATASET_ROOT --seed $SEED
    
done

echo "All experiments completed at $(date)"