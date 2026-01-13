#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --time=10:00:00
#SBATCH --output=/home/users/mespi/projects/triffuser/slurm-%x.%j.out
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH -w gpuhost016
#SBATCH --cpus-per-task=32
#SBATCH --job-name=copy-latents-folder
#SBATCH --mem=450G
##------------------------ End job description ------------------------


# Copy the data.mdb file to the destination folder
# echo "Starting copy of test data.mdb file"
# cp /work/scratch-pw3/mespi/majorTOM/world/latents/test_merged/data.mdb /gws/nopw/j04/sensecdt/data/internal/majorTOM/world/latents/test_merged/data.mdb
# echo "==> Done copying test data.mdb file"

echo "Starting copy of train data.mdb file"
cp /work/scratch-pw3/mespi/majorTOM/world/latents/train_merged/data.mdb /gws/nopw/j04/sensecdt/data/internal/majorTOM/world/latents/train_merged/data.mdb
echo "==> Done copying train data.mdb file"


# Copy the rest of the folder
# rsync -avh --progress /work/scratch-pw3/mespi/majorTOM/world/latents/ /gws/nopw/j04/sensecdt/data/internal/majorTOM/world/latents/
