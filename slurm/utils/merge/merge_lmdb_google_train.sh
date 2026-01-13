#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x.%j.out
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid48
#SBATCH --gres=gpu:4
#SBATCH -x gpuhost015,gpuhost004,gpuhost016
#SBATCH --cpus-per-task=32
#SBATCH --job-name=merge_lmdb_google_train
#SBATCH --mem=400G
##------------------------ End job description ------------------------
module purge && source /gws/nopw/j04/sensecdt/users/mespi/virtualenvs/triffuser/bin/activate

# Copy lmdb files to /tmp NVMe disk
src_dir="/work/scratch-pw3/mespi/majorTOM/world/latents_google_embeds/train"
# dst_dir="/tmp/majorTOM/world/latents_google_embeds/train"
final_dir="/work/scratch-pw3/mespi/majorTOM/world/latents_google_embeds/train_merged"

# Make ln -s to the rest of modalities
# original_dir="/work/scratch-pw3/mespi/majorTOM/world/latents/train"
# for modality in $(ls $original_dir); do
#     if [ ! -L "$src_dir/$modality" ]; then
#         ln -s "$original_dir/$modality" "$src_dir/$modality"
#     fi
# done

# ./slurm/fast_copy_LMDBs.sh $src_dir $dst_dir

# Merge lmdb files
python scripts/merge_lmdbs.py \
    --input_dir $src_dir \
    --output_dir $final_dir \
    --batch_size 20000 \
    --reference_modality "GOOGLE_EMBEDS_GOOGLE_EMBEDS"

echo "Done merging lmdb files!"