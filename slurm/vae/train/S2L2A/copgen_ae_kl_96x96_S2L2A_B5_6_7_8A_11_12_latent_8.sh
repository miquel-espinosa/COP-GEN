#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x.%j.out
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:4
#SBATCH -x gpuhost015,gpuhost004,gpuhost009,gpuhost012
#SBATCH --cpus-per-task=4
#SBATCH --job-name=1node_copgen_ae_kl_96x96_S2L2A_B5_6_7_8A_11_12_latent_8
#SBATCH --mem=400G
##------------------------ End job description ------------------------
module purge && source /gws/nopw/j04/sensecdt/users/mespi/virtualenvs/triffuser/bin/activate

# Export necessary environment variables for distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=12802
export WORLD_SIZE=4  # Total number of GPUs
export NODE_RANK=0   # This node's rank (0 since it's a single node)

accelerate launch \
    --num_processes 4 \
    --multi_gpu \
    train_vae.py \
        --cfg ./configs/copgen/final/S2L2A/copgen_ae_kl_96x96_S2L2A_B5_6_7_8A_11_12_latent_8.yaml