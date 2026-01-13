#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x.%j.out
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:2
#SBATCH -x gpuhost015,gpuhost016,gpuhost004
#SBATCH --cpus-per-task=32
#SBATCH --job-name=encode_copgen_ae_kl_96x96_S2L1C_B5_6_7_8A_11_12_latent_8
#SBATCH --mem=400G
##------------------------ End job description ------------------------
module purge && source /gws/nopw/j04/sensecdt/users/mespi/virtualenvs/triffuser/bin/activate

# Export necessary environment variables for distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=12802
export WORLD_SIZE=1  # Total number of GPUs
export NODE_RANK=0   # This node's rank (0 since it's a single node)

# export SOURCE_DIR="/gws/nopw/j04/sensecdt/data/internal/majorTOM/world/Core-S2L1C-tar-gz"
export SOURCE_DIR="/work/scratch-pw3/mespi/majorTOM/world/Core-S2L1C-tar-gz"
export TMP_DIR="/tmp/majorTOM/world/Core-S2L1C"
export TMP_TARS_DIR="$TMP_DIR-tars"

echo "We are using the compute node $HOSTNAME!"

# Copy the data to /tmp
python scripts/copy_extract.py \
  --source-dir $SOURCE_DIR \
  --tmp-tars-dir $TMP_TARS_DIR \
  --dest-dir $TMP_DIR \
  --num-workers 32

echo "Done copying data to /tmp!"



echo "Starting LMDB dataset encoding..."

accelerate launch --num_processes 1 encode_moments_vae.py \
  --cfg ./configs/copgen/final/S2L1C/copgen_ae_kl_96x96_S2L1C_B5_6_7_8A_11_12_latent_8.yaml \
  --checkpoint_path ./models/S2L1C_96x96_B5_6_7_8A_11_12_latent_8/model-50.pt \
  --output_dir /work/scratch-pw3/mespi/majorTOM/world/latents \
  --batch_size 64 \
  --patchify \
  --latents_only \
  --num_workers 32 \
  --lulc_align \
  --save_to_lmdb --lmdb_fast