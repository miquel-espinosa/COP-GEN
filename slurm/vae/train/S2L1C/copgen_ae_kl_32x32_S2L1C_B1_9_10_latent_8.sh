#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x.%j.out
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH -x gpuhost015,gpuhost004
#SBATCH --cpus-per-task=4
#SBATCH --job-name=1node_copgen_ae_kl_32x32_S2L1C_B1_9_10_latent_8
#SBATCH --mem=200G
##------------------------ End job description ------------------------
module purge && source /gws/nopw/j04/sensecdt/users/mespi/virtualenvs/triffuser/bin/activate

# Export necessary environment variables for distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=12802
export WORLD_SIZE=1  # Total number of GPUs
export NODE_RANK=0   # This node's rank (0 since it's a single node)

accelerate launch \
    --num_processes 1 \
    train_vae.py \
        --cfg ./configs/copgen/final/S2L1C/copgen_ae_kl_32x32_S2L1C_B1_9_10_latent_8.yaml
    # --num_processes 4 \
    # --multi_gpu \