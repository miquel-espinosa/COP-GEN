#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --time=48:00:00
#SBATCH --output=/home/users/mespi/projects/triffuser/slurm-%x.%j.out
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid48
#SBATCH --gres=gpu:4
#SBATCH -x gpuhost015,gpuhost016
#SBATCH --cpus-per-task=32
#SBATCH --job-name=train-4gpu-copgen-base
#SBATCH --mem=450G
##------------------------ End job description ------------------------
####SBATCH --job-name=train-copgen-world-12-modalities
module purge && source /gws/nopw/j04/sensecdt/users/mespi/virtualenvs/triffuser/bin/activate
export CACHE_DIR=/tmp
export NUM_GPUS=4

# =========================COPY LMDB TO TMP FILE=========================

# # Copy lmdb to tmp file in the compute node and time the operation
# export LMDB_TMP_DIR=/tmp/majorTOM/world/latents/train_merged
# export LMDB_DIR=/work/scratch-pw3/mespi/majorTOM/world/latents/train_merged
# mkdir -p $LMDB_TMP_DIR

# echo "Starting copy of lock.mdb to $LMDB_TMP_DIR"
# start_time_lock=$(date +%s)
# cp -r $LMDB_DIR/lock.mdb $LMDB_TMP_DIR/lock.mdb
# end_time_lock=$(date +%s)
# echo "Copied lock.mdb to $LMDB_TMP_DIR in $((end_time_lock - start_time_lock)) seconds"

# echo "Starting copy of data.mdb to $LMDB_TMP_DIR"
# start_time_data=$(date +%s)
# cp -r $LMDB_DIR/data.mdb $LMDB_TMP_DIR/data.mdb
# end_time_data=$(date +%s)
# echo "Copied data.mdb to $LMDB_TMP_DIR in $((end_time_data - start_time_data)) seconds"

# =========================TRAIN COPGEN=========================

# Make current path the project root
cd /home/users/mespi/projects/triffuser

# Print the name of the host
export HOSTNAME=$(hostname)
echo $HOSTNAME

# Clean up checkpoints
./slurm/clean_checkpoints.sh workdir/copgen/discrete/cop_gen_base/default/ckpts

export OMP_NUM_THREADS=4
accelerate launch \
            --multi_gpu \
            --num_processes 4 \
            --mixed_precision fp16 \
            train_copgen_discrete.py \
                --config="configs/copgen/discrete/cop_gen_base.py"
