#!/bin/bash
# filepath: /home/egm/Data/Projects/CopGen/leave-one-out-experiments.sh

echo "Starting generations..."
echo "Working directory: $(pwd)"
echo "Timestamp: $(date)"
echo ""

# Modalities: DEM LULC S1RTC S2L2A S2L1C coords

DATASET_ROOT="./data/majorTOM/test_dataset_copgen_leave_one_out"
MODEL_PATH="./models/copgen/cop_gen_base/500000_nnet_ema.pth"
CONFIG_PATH="./configs/copgen/discrete/cop_gen_base.py"
# MODEL_PATH="./models/copgen/cop_gen_huge/273000_nnet_ema.pth"
# CONFIG_PATH="./configs/copgen/discrete/cop_gen_huge.py"
# SEEDS=(111 222 333 444 555 666 777 888)
SEEDS=(12 23 34 45 56 67 78 89)
BATCH_SIZE=64
VIS_EVERY=2

for SEED in ${SEEDS[@]}; do
    echo -e "\033[31mRunning experiment with seed $SEED\033[0m"

    # ------------------------DEM---------------------------------
    echo -e "\033[31mTarget: DEM\033[0m"
    # Remove coords
    python3 copgen_cli.py --dataset-root $DATASET_ROOT --model $MODEL_PATH --config $CONFIG_PATH --seed $SEED \
                        --inputs LULC S1RTC S2L1C S2L2A cloud_mask \
                        --batch-size $BATCH_SIZE --vis-every $VIS_EVERY

    # Remove LULC
    python3 copgen_cli.py --dataset-root $DATASET_ROOT --model $MODEL_PATH --config $CONFIG_PATH --seed $SEED \
                        --inputs lat_lon S1RTC S2L1C S2L2A cloud_mask \
                        --batch-size $BATCH_SIZE --vis-every $VIS_EVERY

    # Remove S1RTC
    python3 copgen_cli.py --dataset-root $DATASET_ROOT --model $MODEL_PATH --config $CONFIG_PATH --seed $SEED \
                        --inputs lat_lon LULC S2L1C S2L2A cloud_mask \
                        --batch-size $BATCH_SIZE --vis-every $VIS_EVERY

    # Remove S2L1C
    python3 copgen_cli.py --dataset-root $DATASET_ROOT --model $MODEL_PATH --config $CONFIG_PATH --seed $SEED \
                        --inputs lat_lon LULC S1RTC S2L2A cloud_mask \
                        --batch-size $BATCH_SIZE --vis-every $VIS_EVERY

    # Remove S2L2A
    python3 copgen_cli.py --dataset-root $DATASET_ROOT --model $MODEL_PATH --config $CONFIG_PATH --seed $SEED \
                        --inputs lat_lon LULC S1RTC S2L1C cloud_mask \
                        --batch-size $BATCH_SIZE --vis-every $VIS_EVERY


    # ------------------------LULC---------------------------------
    echo -e "\033[31mTarget: LULC\033[0m"

    # Remove coords
    python3 copgen_cli.py --dataset-root $DATASET_ROOT --model $MODEL_PATH --config $CONFIG_PATH --seed $SEED \
                        --inputs DEM S1RTC S2L1C S2L2A cloud_mask \
                        --batch-size $BATCH_SIZE --vis-every $VIS_EVERY

    # Remove S1RTC
    python3 copgen_cli.py --dataset-root $DATASET_ROOT --model $MODEL_PATH --config $CONFIG_PATH --seed $SEED \
                        --inputs lat_lon DEM S2L1C S2L2A cloud_mask \
                        --batch-size $BATCH_SIZE --vis-every $VIS_EVERY

    # Remove S2L2A
    python3 copgen_cli.py --dataset-root $DATASET_ROOT --model $MODEL_PATH --config $CONFIG_PATH --seed $SEED \
                        --inputs lat_lon DEM S1RTC S2L1C cloud_mask \
                        --batch-size $BATCH_SIZE --vis-every $VIS_EVERY
    
    # Remove S2L1C
    python3 copgen_cli.py --dataset-root $DATASET_ROOT --model $MODEL_PATH --config $CONFIG_PATH --seed $SEED \
                        --inputs lat_lon DEM S1RTC S2L2A cloud_mask \
                        --batch-size $BATCH_SIZE --vis-every $VIS_EVERY


    # ------------------------S1RTC---------------------------------
    echo -e "\033[31mTarget: S1RTC\033[0m"

    # Remove coords
    python3 copgen_cli.py --dataset-root $DATASET_ROOT --model $MODEL_PATH --config $CONFIG_PATH --seed $SEED \
                        --inputs DEM LULC S2L1C S2L2A cloud_mask \
                        --batch-size $BATCH_SIZE --vis-every $VIS_EVERY

    # Remove S2L2A
    python3 copgen_cli.py --dataset-root $DATASET_ROOT --model $MODEL_PATH --config $CONFIG_PATH --seed $SEED \
                        --inputs lat_lon DEM LULC S2L1C cloud_mask \
                        --batch-size $BATCH_SIZE --vis-every $VIS_EVERY

    # Remove S2L1C
    python3 copgen_cli.py --dataset-root $DATASET_ROOT --model $MODEL_PATH --config $CONFIG_PATH --seed $SEED \
                        --inputs lat_lon DEM LULC S2L2A cloud_mask \
                        --batch-size $BATCH_SIZE --vis-every $VIS_EVERY



    # ------------------------S2L1C---------------------------------
    echo -e "\033[31mTarget: S2L1C\033[0m"

    # Remove coords
    python3 copgen_cli.py --dataset-root $DATASET_ROOT --model $MODEL_PATH --config $CONFIG_PATH --seed $SEED \
                        --inputs DEM LULC S1RTC S2L2A cloud_mask \
                        --batch-size $BATCH_SIZE --vis-every $VIS_EVERY

    # Remove S2L2A
    python3 copgen_cli.py --dataset-root $DATASET_ROOT --model $MODEL_PATH --config $CONFIG_PATH --seed $SEED \
                        --inputs lat_lon DEM LULC S1RTC cloud_mask \
                        --batch-size $BATCH_SIZE --vis-every $VIS_EVERY



    # ------------------------S2L2A---------------------------------
    echo -e "\033[31mTarget: S2L2A\033[0m"

    # Remove coords
    python3 copgen_cli.py --dataset-root $DATASET_ROOT --model $MODEL_PATH --config $CONFIG_PATH --seed $SEED \
                        --inputs DEM LULC S1RTC S2L1C cloud_mask \
                        --batch-size $BATCH_SIZE --vis-every $VIS_EVERY

done

echo "All experiments completed at $(date)"