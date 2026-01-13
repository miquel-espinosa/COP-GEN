#!/bin/bash
# filepath: /home/egm/Data/Projects/CopGen/leave-one-out-experiments.sh

echo "Starting generations..."
echo "Working directory: $(pwd)"
echo "Timestamp: $(date)"
echo ""

# Available modalities: lat_lon, DEM, LULC, S1RTC, S2L1C, S2L2A, cloud_mask, timestamps

DATASET_ROOT="./data/majorTOM/test_dataset_copgen"
MODEL_PATH="./models/copgen/cop_gen_base/500000_nnet_ema.pth"
CONFIG_PATH="./configs/copgen/discrete/cop_gen_base.py"
# CONFIG_PATH="./configs/copgen/discrete/cop_gen_base.py"
# MODEL_PATH="./models/copgen/cop_gen_huge/273000_nnet_ema.pth"
# CONFIG_PATH="./configs/copgen/discrete/cop_gen_huge.py"
SEED_START=20
SEED_END=39
BATCH_SIZE=16
VIS_EVERY=8

for SEED in $(seq $SEED_START $SEED_END); do
    echo -e "\033[31mRunning experiment with seed $SEED\033[0m"

    # ------------------------lat_lon---------------------------------
    echo -e "\033[31mTarget: lat_lon\033[0m"
    python3 copgen_cli.py --dataset-root $DATASET_ROOT --model $MODEL_PATH --config $CONFIG_PATH --seed $SEED \
                        --inputs DEM LULC S1RTC S2L1C S2L2A cloud_mask timestamps \
                        --batch-size $BATCH_SIZE --vis-every $VIS_EVERY

    # ------------------------DEM---------------------------------
    echo -e "\033[31mTarget: DEM\033[0m"
    python3 copgen_cli.py --dataset-root $DATASET_ROOT --model $MODEL_PATH --config $CONFIG_PATH --seed $SEED \
                        --inputs lat_lon LULC S1RTC S2L1C S2L2A cloud_mask timestamps \
                        --batch-size $BATCH_SIZE --vis-every $VIS_EVERY

    # ------------------------LULC---------------------------------
    echo -e "\033[31mTarget: LULC\033[0m"
    python3 copgen_cli.py --dataset-root $DATASET_ROOT --model $MODEL_PATH --config $CONFIG_PATH --seed $SEED \
                        --inputs lat_lon DEM S1RTC S2L1C S2L2A cloud_mask timestamps \
                        --batch-size $BATCH_SIZE --vis-every $VIS_EVERY

    # ------------------------S1RTC---------------------------------
    echo -e "\033[31mTarget: S1RTC\033[0m"
    python3 copgen_cli.py --dataset-root $DATASET_ROOT --model $MODEL_PATH --config $CONFIG_PATH --seed $SEED \
                        --inputs lat_lon DEM LULC S2L1C S2L2A cloud_mask timestamps \
                        --batch-size $BATCH_SIZE --vis-every $VIS_EVERY

    # ------------------------S2L1C---------------------------------
    echo -e "\033[31mTarget: S2L1C\033[0m"
    python3 copgen_cli.py --dataset-root $DATASET_ROOT --model $MODEL_PATH --config $CONFIG_PATH --seed $SEED \
                        --inputs lat_lon DEM LULC S1RTC S2L2A cloud_mask timestamps \
                        --batch-size $BATCH_SIZE --vis-every $VIS_EVERY

    # ------------------------S2L2A---------------------------------
    echo -e "\033[31mTarget: S2L2A\033[0m"
    python3 copgen_cli.py --dataset-root $DATASET_ROOT --model $MODEL_PATH --config $CONFIG_PATH --seed $SEED \
                        --inputs lat_lon DEM LULC S1RTC S2L1C cloud_mask timestamps \
                        --batch-size $BATCH_SIZE --vis-every $VIS_EVERY

    # ------------------------S2 all---------------------------------
    echo -e "\033[31mTarget: S2 all\033[0m"
    python3 copgen_cli.py --dataset-root $DATASET_ROOT --model $MODEL_PATH --config $CONFIG_PATH --seed $SEED \
                        --inputs lat_lon DEM LULC S1RTC cloud_mask timestamps \
                        --batch-size $BATCH_SIZE --vis-every $VIS_EVERY
done

echo "All experiments completed at $(date)"