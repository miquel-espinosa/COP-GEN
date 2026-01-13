#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x.%j.out
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --gres=gpu:4
#SBATCH -w gpuhost011
#SBATCH --cpus-per-task=28
#SBATCH --job-name=generate-test-dem-s2l1c
#SBATCH --mem=400G
##------------------------ End job description ------------------------
module purge && source /gws/nopw/j04/sensecdt/users/mespi/virtualenvs/triffuser/bin/activate

MODEL_PATH="workdir/majortom/discrete/lmdb/world_dems1s2s2_triffuser/batch_size=2048/ckpts/114000.ckpt/nnet_ema.pth"
DATA_PATH="data/majorTOM/world/world_thumbnail_centercrop_npy_lmdb_4modalities/test"
OUT_PATH_MAIN_FOLDER="work_scratch_out_images"

# TEST SET 1. Condition on DEM
CUDA_VISIBLE_DEVICES=0 python3 sample_n_triffuser.py \
    --config configs/majortom/discrete/lmdb/world_dems1s2s2_triffuser.py \
    --data_path $DATA_PATH \
    --data_type lmdb \
    --nnet_path $MODEL_PATH \
    --n_mod 4 \
    --condition dem \
    --generate s1_rtc,s2_l2a,s2_l1c \
    --output_path $OUT_PATH_MAIN_FOLDER/test_set_1 \
    --save_as pngs &

# TEST SET 1. Condition on S2 L1C
CUDA_VISIBLE_DEVICES=1 python3 sample_n_triffuser.py \
    --config configs/majortom/discrete/lmdb/world_dems1s2s2_triffuser.py \
    --data_path $DATA_PATH \
    --data_type lmdb \
    --nnet_path $MODEL_PATH \
    --n_mod 4 \
    --condition s2_l1c \
    --generate dem,s1_rtc,s2_l2a \
    --output_path $OUT_PATH_MAIN_FOLDER/test_set_1 \
    --save_as pngs &


## REPEAT THE ABOVE FOR TEST SET 2

# TEST SET 2. Condition on DEM
CUDA_VISIBLE_DEVICES=0 python3 sample_n_triffuser.py \
    --config configs/majortom/discrete/lmdb/world_dems1s2s2_triffuser.py \
    --data_path $DATA_PATH \
    --data_type lmdb \
    --nnet_path $MODEL_PATH \
    --n_mod 4 \
    --condition dem \
    --generate s1_rtc,s2_l2a,s2_l1c \
    --output_path $OUT_PATH_MAIN_FOLDER/test_set_2 \
    --save_as pngs &

# TEST SET 2. Condition on S2 L1C
CUDA_VISIBLE_DEVICES=1 python3 sample_n_triffuser.py \
    --config configs/majortom/discrete/lmdb/world_dems1s2s2_triffuser.py \
    --data_path $DATA_PATH \
    --data_type lmdb \
    --nnet_path $MODEL_PATH \
    --n_mod 4 \
    --condition s2_l1c \
    --generate dem,s1_rtc,s2_l2a \
    --output_path $OUT_PATH_MAIN_FOLDER/test_set_2 \
    --save_as pngs &


wait


# TEST SET 3. Condition on DEM
CUDA_VISIBLE_DEVICES=0 python3 sample_n_triffuser.py \
    --config configs/majortom/discrete/lmdb/world_dems1s2s2_triffuser.py \
    --data_path $DATA_PATH \
    --data_type lmdb \
    --nnet_path $MODEL_PATH \
    --n_mod 4 \
    --condition dem \
    --generate s1_rtc,s2_l2a,s2_l1c \
    --output_path $OUT_PATH_MAIN_FOLDER/test_set_3 \
    --save_as pngs &

# TEST SET 3. Condition on S2 L1C
CUDA_VISIBLE_DEVICES=1 python3 sample_n_triffuser.py \
    --config configs/majortom/discrete/lmdb/world_dems1s2s2_triffuser.py \
    --data_path $DATA_PATH \
    --data_type lmdb \
    --nnet_path $MODEL_PATH \
    --n_mod 4 \
    --condition s2_l1c \
    --generate dem,s1_rtc,s2_l2a \
    --output_path $OUT_PATH_MAIN_FOLDER/test_set_3 \
    --save_as pngs &


## REPEAT THE ABOVE FOR TEST SET 4

# TEST SET 4. Condition on DEM
CUDA_VISIBLE_DEVICES=0 python3 sample_n_triffuser.py \
    --config configs/majortom/discrete/lmdb/world_dems1s2s2_triffuser.py \
    --data_path $DATA_PATH \
    --data_type lmdb \
    --nnet_path $MODEL_PATH \
    --n_mod 4 \
    --condition dem \
    --generate s1_rtc,s2_l2a,s2_l1c \
    --output_path $OUT_PATH_MAIN_FOLDER/test_set_4 \
    --save_as pngs &

# TEST SET 4. Condition on S2 L1C
CUDA_VISIBLE_DEVICES=1 python3 sample_n_triffuser.py \
    --config configs/majortom/discrete/lmdb/world_dems1s2s2_triffuser.py \
    --data_path $DATA_PATH \
    --data_type lmdb \
    --nnet_path $MODEL_PATH \
    --n_mod 4 \
    --condition s2_l1c \
    --generate dem,s1_rtc,s2_l2a \
    --output_path $OUT_PATH_MAIN_FOLDER/test_set_4 \
    --save_as pngs &

wait