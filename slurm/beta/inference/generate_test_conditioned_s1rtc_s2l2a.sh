#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x.%j.out
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --gres=gpu:4
#SBATCH -w gpuhost011
#SBATCH --cpus-per-task=28
#SBATCH --job-name=generate-test-v2
#SBATCH --mem=400G
##------------------------ End job description ------------------------
module purge && source /gws/nopw/j04/sensecdt/users/mespi/virtualenvs/triffuser/bin/activate

MODEL_PATH="workdir/majortom/discrete/lmdb/world_dems1s2s2_triffuser/batch_size=2048/ckpts/114000.ckpt/nnet_ema.pth"
DATA_PATH="data/majorTOM/world/world_thumbnail_centercrop_npy_lmdb_4modalities/test"
OUT_PATH_MAIN_FOLDER="work_scratch_out_images"

# TEST SET 3. Condition on S1 RTC
CUDA_VISIBLE_DEVICES=0 python3 sample_n_triffuser.py \
    --config configs/majortom/discrete/lmdb/world_dems1s2s2_triffuser.py \
    --data_path $DATA_PATH \
    --data_type lmdb \
    --nnet_path $MODEL_PATH \
    --n_mod 4 \
    --condition s1_rtc \
    --generate dem,s2_l2a,s2_l1c \
    --output_path $OUT_PATH_MAIN_FOLDER/test_set_3 \
    --save_as pngs &

# TEST SET 3. Condition on S2 L2A
CUDA_VISIBLE_DEVICES=1 python3 sample_n_triffuser.py \
    --config configs/majortom/discrete/lmdb/world_dems1s2s2_triffuser.py \
    --data_path $DATA_PATH \
    --data_type lmdb \
    --nnet_path $MODEL_PATH \
    --n_mod 4 \
    --condition s2_l2a \
    --generate dem,s1_rtc,s2_l1c \
    --output_path $OUT_PATH_MAIN_FOLDER/test_set_3 \
    --save_as pngs &


## REPEAT THE ABOVE FOR TEST SET 4

# TEST SET 4. Condition on S1 RTC
CUDA_VISIBLE_DEVICES=2 python3 sample_n_triffuser.py \
    --config configs/majortom/discrete/lmdb/world_dems1s2s2_triffuser.py \
    --data_path $DATA_PATH \
    --data_type lmdb \
    --nnet_path $MODEL_PATH \
    --n_mod 4 \
    --condition s1_rtc \
    --generate dem,s2_l2a,s2_l1c \
    --output_path $OUT_PATH_MAIN_FOLDER/test_set_4 \
    --save_as pngs &

# TEST SET 4. Condition on S2 L2A
CUDA_VISIBLE_DEVICES=3 python3 sample_n_triffuser.py \
    --config configs/majortom/discrete/lmdb/world_dems1s2s2_triffuser.py \
    --data_path $DATA_PATH \
    --data_type lmdb \
    --nnet_path $MODEL_PATH \
    --n_mod 4 \
    --condition s2_l2a \
    --generate dem,s1_rtc,s2_l1c \
    --output_path $OUT_PATH_MAIN_FOLDER/test_set_4 \
    --save_as pngs &

wait