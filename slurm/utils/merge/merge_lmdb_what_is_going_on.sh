#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x.%j.out
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:4
#SBATCH -x gpuhost015,gpuhost004,gpuhost016
#SBATCH --cpus-per-task=32
#SBATCH --job-name=merge_lmdb_what_is_going_on
#SBATCH --mem=400G
##------------------------ End job description ------------------------
module purge && source /gws/nopw/j04/sensecdt/users/mespi/virtualenvs/triffuser/bin/activate

python scripts/merge_lmdbs_what_is_going_on.py \
    --input_dir /work/scratch-pw3/mespi/majorTOM/world/latents/train \
    --output_dir /work/scratch-pw3/mespi/majorTOM/world/latents/train_merged \
    --num_workers 32



