#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x.%j.out
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH -x gpuhost015,gpuhost016
#SBATCH --cpus-per-task=16
#SBATCH --job-name=compute-scaling-factors
#SBATCH --mem=400G
##------------------------ End job description ------------------------
module purge && source /gws/nopw/j04/sensecdt/users/mespi/virtualenvs/triffuser/bin/activate


# echo -e "\033[1;33mSCALING FACTORS WITH 50 SAMPLES\033[0m"
# python compute_scaling_factor_simplified.py \
#   --latent_path /work/scratch-pw3/mespi/majorTOM/world/latents/train \
#   --config_path ./configs/copgen/discrete/world_12_modalities.py \
#   --grid_cells_file /work/scratch-pw3/mespi/majorTOM/world/latents/train.txt \
#   --num_samples 50 \
#   --batch_size 50

# echo -e "\033[1;33mSCALING FACTORS WITH 500 SAMPLES\033[0m"
# python compute_scaling_factor_simplified.py \
#   --latent_path /work/scratch-pw3/mespi/majorTOM/world/latents/train \
#   --config_path ./configs/copgen/discrete/world_12_modalities.py \
#   --grid_cells_file /work/scratch-pw3/mespi/majorTOM/world/latents/train.txt \
#   --num_samples 500 \
#   --batch_size 500

echo -e "\033[1;33mSCALING FACTORS WITH 5000 SAMPLES\033[0m"
python scripts/compute_scaling_factor_simplified.py \
  --latent_path /work/scratch-pw3/mespi/majorTOM/world/latents/train \
  --config_path ./configs/copgen/discrete/world_12_modalities.py \
  --grid_cells_file /work/scratch-pw3/mespi/majorTOM/world/latents/train.txt \
  --num_samples 5000 \
  --batch_size 5000

  
echo -e "\033[1;33mSCALING FACTORS WITH 5000 SAMPLES, 2ND RUN\033[0m"
python scripts/compute_scaling_factor_simplified.py \
  --latent_path /work/scratch-pw3/mespi/majorTOM/world/latents/train \
  --config_path ./configs/copgen/discrete/world_12_modalities.py \
  --grid_cells_file /work/scratch-pw3/mespi/majorTOM/world/latents/train.txt \
  --num_samples 5000 \
  --batch_size 5000



