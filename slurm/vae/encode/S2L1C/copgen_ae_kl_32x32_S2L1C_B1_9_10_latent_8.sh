#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x.%j.out
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:4
#SBATCH -x gpuhost015,gpuhost016,gpuhost012
#SBATCH --cpus-per-task=32
#SBATCH --job-name=encode_copgen_ae_kl_32x32_S2L1C_B1_9_10_latent_8
#SBATCH --mem=400G
##------------------------ End job description ------------------------
module purge && source /gws/nopw/j04/sensecdt/users/mespi/virtualenvs/triffuser/bin/activate

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

export TMP_DIR_LMDB="/tmp/majorTOM/world/latents"
export DEST_DIR_LMDB="/work/scratch-pw3/mespi/majorTOM/world/latents"
mkdir -p $TMP_DIR_LMDB

# Copy train.txt and test.txt to tmp dir
cp $DEST_DIR_LMDB/train.txt $TMP_DIR_LMDB/
cp $DEST_DIR_LMDB/test.txt $TMP_DIR_LMDB/

accelerate launch --multi_gpu --num_processes 4 encode_moments_vae.py \
  --cfg ./configs/copgen/final/S2L1C/copgen_ae_kl_32x32_S2L1C_B1_9_10_latent_8.yaml \
  --checkpoint_path ./models/S2L1C_32x32_B1_9_10_latent_8/model-50.pt \
  --output_dir $TMP_DIR_LMDB \
  --batch_size 128 \
  --patchify \
  --latents_only \
  --num_workers 32 \
  --lulc_align \
  --save_to_lmdb --lmdb_fast

echo "Done encoding LMDB dataset!"

# Copy tmp lmdb to destination with cp command
cp -r $TMP_DIR_LMDB/train/* $DEST_DIR_LMDB/train/
cp -r $TMP_DIR_LMDB/test/* $DEST_DIR_LMDB/test/

echo "Done copying LMDB dataset to destination!"