#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x.%j.out
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --job-name=install
#SBATCH --mem=400G
##------------------------ End job description ------------------------
module purge && source /gws/nopw/j04/sensecdt/users/mespi/virtualenvs/triffuser/bin/activate

module load gcc/8.2.0
export CUDA_HOME=/usr/local/cuda-11.7

# If using conda: conda create -n triffuser python=3.11 -y
pip install torch torchvision accelerate absl-py ml_collections einops wandb ftfy transformers
pip install xformers
pip install --pre triton
pip install timm
pip install scipy tqdm
pip install pandas
pip install leafmap datasets
pip install geopandas rasterio shapely fsspec pyarrow matplotlib
pip install seaborn
pip install basemap

# To solve error:
#   [rank1]:[W220 08:50:06.072103879 ProcessGroupNCCL.cpp:4115] [PG ID 0 PG GUID 0 Rank 1] 
#   using GPU 1 to perform barrier as devices used by this process are currently unknown.
#   This can potentially cause a hang if this rank to GPU mapping is incorrect.Specify device_ids in barrier()
#   to force use of a particular device,or call init_process_group() with a device_id.
# We need to downgrade pytorch and xformers and CUDA to 12.1
# pip install xformers==0.0.28.dev895
# pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
