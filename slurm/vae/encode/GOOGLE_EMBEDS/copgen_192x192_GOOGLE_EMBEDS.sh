#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x.%j.out
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid48
#SBATCH --gres=gpu:4
#SBATCH -x gpuhost015,gpuhost016,gpuhost010
#SBATCH --cpus-per-task=32
#SBATCH --job-name=encode_copgen_ae_kl_192x192_GOOGLE_EMBEDS
#SBATCH --mem=400G
##------------------------ End job description ------------------------
module purge && source /gws/nopw/j04/sensecdt/users/mespi/virtualenvs/triffuser/bin/activate

# Export necessary environment variables for distributed training
export MASTER_ADDR=$(hostname)
export MASTER_PORT=12802
export WORLD_SIZE=1  # Total number of GPUs
export NODE_RANK=0   # This node's rank (0 since it's a single node)

export SOURCE_DIR="/work/scratch-pw3/mespi/majorTOM/world/Core-GOOGLE_EMBEDS-tar-gz"
export TMP_DIR="/tmp/majorTOM/world/Core-GOOGLE_EMBEDS"
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

# export DEST_DIR_LMDB="/work/scratch-pw3/mespi/majorTOM/world/latents"
export DEST_DIR_LMDB="/work/scratch-pw3/mespi/majorTOM/world/latents_google_embeds"

accelerate launch --multi_gpu --num_processes 4 encode_moments_vae.py \
  --cfg ./configs/copgen/final/GOOGLE_EMBEDS/copgen_192x192_GOOGLE_EMBEDS.yaml \
  --output_dir $DEST_DIR_LMDB \
  --batch_size 16 \
  --patchify \
  --latents_only \
  --num_workers 32 \
  --lulc_align \
  --save_to_lmdb \
  --lmdb_fast \
  --flush_every_batches 16 \
  --save_to_zarr \
  --resume

# --lmdb_map_size 19791209299968 \
# 18TB = 19791209299968 bytes

echo "Done encoding LMDB dataset!"

# Copy tmp lmdb to destination with cp command
# cp -r $TMP_DIR_LMDB/train/* $DEST_DIR_LMDB/train/
# cp -r $TMP_DIR_LMDB/test/* $DEST_DIR_LMDB/test/

# echo "Done copying LMDB dataset to destination!"