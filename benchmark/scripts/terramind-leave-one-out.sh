#!/bin/bash
# filepath: /home/egm/Data/Projects/CopGen/leave-one-out-experiments.sh

echo "Starting generations..."
echo "Working directory: $(pwd)"
echo "Timestamp: $(date)"
echo ""

DATASET_ROOT="./data/majorTOM/test_dataset_copgen_leave_one_out"
# SEEDS=(111 222 333 444 555 666 777 888)
SEEDS=(12 23 34 45 56 67 78 89)

# Modalities: DEM LULC S1RTC S2L2A S2L1C coords

for SEED in ${SEEDS[@]}; do
    echo -e "\033[31mRunning experiment with seed $SEED\033[0m"

    # target: DEM
    python terramind_cli.py --input LULC S1RTC S2L2A S2L1C   --output DEM --dataset_root $DATASET_ROOT --seed $SEED # removed: coords
    python terramind_cli.py --input coords S1RTC S2L2A S2L1C --output DEM --dataset_root $DATASET_ROOT --seed $SEED # removed: LULC
    python terramind_cli.py --input coords LULC S2L2A S2L1C  --output DEM --dataset_root $DATASET_ROOT --seed $SEED # removed: S1RTC
    python terramind_cli.py --input coords LULC S1RTC S2L1C  --output DEM --dataset_root $DATASET_ROOT --seed $SEED # removed: S2L2A
    python terramind_cli.py --input coords LULC S1RTC S2L2A  --output DEM --dataset_root $DATASET_ROOT --seed $SEED # removed: S2L1C

    # target: LULC
    python terramind_cli.py --input DEM S1RTC S2L2A S2L1C    --output LULC --dataset_root $DATASET_ROOT --seed $SEED # removed: coords
    python terramind_cli.py --input DEM coords S2L2A S2L1C   --output LULC --dataset_root $DATASET_ROOT --seed $SEED # removed: S1RTC
    python terramind_cli.py --input DEM coords S1RTC S2L1C   --output LULC --dataset_root $DATASET_ROOT --seed $SEED # removed: S2L2A
    python terramind_cli.py --input DEM coords S1RTC S2L2A   --output LULC --dataset_root $DATASET_ROOT --seed $SEED # removed: S2L1C
    python terramind_cli.py --input coords S1RTC S2L2A S2L1C --output LULC --dataset_root $DATASET_ROOT --seed $SEED # removed: DEM

    # target: S1RTC
    python terramind_cli.py --input DEM LULC S2L2A S2L1C    --output S1RTC --dataset_root $DATASET_ROOT --seed $SEED # removed: coords
    python terramind_cli.py --input DEM LULC coords S2L1C   --output S1RTC --dataset_root $DATASET_ROOT --seed $SEED # removed: S2L2A
    python terramind_cli.py --input DEM LULC coords S2L2A   --output S1RTC --dataset_root $DATASET_ROOT --seed $SEED # removed: S2L1C
    python terramind_cli.py --input coords LULC S2L2A S2L1C --output S1RTC --dataset_root $DATASET_ROOT --seed $SEED # removed: DEM
    python terramind_cli.py --input DEM coords S2L2A S2L1C  --output S1RTC --dataset_root $DATASET_ROOT --seed $SEED # removed: LULC

    # target: S2L1C
    python terramind_cli.py --input DEM LULC S1RTC S2L2A    --output S2L1C --dataset_root $DATASET_ROOT --seed $SEED # removed: coords
    python terramind_cli.py --input DEM LULC coords S2L2A   --output S2L1C --dataset_root $DATASET_ROOT --seed $SEED # removed: S1RTC
    python terramind_cli.py --input DEM LULC coords S1RTC   --output S2L1C --dataset_root $DATASET_ROOT --seed $SEED # removed: S2L2A
    python terramind_cli.py --input coords LULC S1RTC S2L2A --output S2L1C --dataset_root $DATASET_ROOT --seed $SEED # removed: DEM
    python terramind_cli.py --input DEM coords S1RTC S2L2A  --output S2L1C --dataset_root $DATASET_ROOT --seed $SEED # removed: LULC

    # target: S2L2A
    python terramind_cli.py --input DEM LULC S1RTC S2L1C    --output S2L2A --dataset_root $DATASET_ROOT --seed $SEED # removed: coords
    python terramind_cli.py --input DEM LULC coords S2L1C   --output S2L2A --dataset_root $DATASET_ROOT --seed $SEED # removed: S1RTC    
    python terramind_cli.py --input DEM LULC coords S1RTC   --output S2L2A --dataset_root $DATASET_ROOT --seed $SEED # removed: S2L1C
    python terramind_cli.py --input coords LULC S1RTC S2L1C --output S2L2A --dataset_root $DATASET_ROOT --seed $SEED # removed: DEM
    python terramind_cli.py --input DEM coords S1RTC S2L1C  --output S2L2A --dataset_root $DATASET_ROOT --seed $SEED # removed: LULC    

    # target: coords
    python terramind_cli.py --input DEM S1RTC S2L2A S2L1C  --output coords --dataset_root $DATASET_ROOT --seed $SEED # removed: LULC
    python terramind_cli.py --input LULC S1RTC S2L2A S2L1C --output coords --dataset_root $DATASET_ROOT --seed $SEED # removed: DEM
    python terramind_cli.py --input DEM LULC S2L2A S2L1C   --output coords --dataset_root $DATASET_ROOT --seed $SEED # removed: S1RTC
    python terramind_cli.py --input DEM LULC S1RTC S2L2A   --output coords --dataset_root $DATASET_ROOT --seed $SEED # removed: S2L1C
    python terramind_cli.py --input DEM LULC S1RTC S2L1C   --output coords --dataset_root $DATASET_ROOT --seed $SEED # removed: S2L2A

done

echo "All experiments completed at $(date)"