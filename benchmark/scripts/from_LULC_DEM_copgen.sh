#!/bin/bash
# filepath: /home/egm/Data/Projects/CopGen/leave-one-out-experiments.sh

echo "Starting generations..."
echo "Working directory: $(pwd)"
echo "Timestamp: $(date)"
echo ""

# Available modalities: lat_lon, DEM, LULC, S1RTC, S2L1C, S2L2A, cloud_mask, timestamps

DATASET_ROOT="./data/majorTOM/test_dataset_copgen_from_DEM_LULC"
MODEL_PATH="./models/copgen/cop_gen_base/500000_nnet_ema.pth"
CONFIG_PATH="./configs/copgen/discrete/cop_gen_base.py"
SEED_START=0
SEED_END=15
BATCH_SIZE=16
VIS_EVERY=8

for SEED in $(seq $SEED_START $SEED_END); do
    echo -e "\033[31mRunning experiment with seed $SEED\033[0m"

    echo -e "\033[31mGenerate all modalities from DEM and LULC\033[0m"
    python3 copgen_cli.py --dataset-root $DATASET_ROOT --model $MODEL_PATH --config $CONFIG_PATH --seed $SEED \
                        --inputs DEM LULC cloud_mask \
                        --batch-size $BATCH_SIZE --vis-every $VIS_EVERY
done

echo "Completed at $(date)"