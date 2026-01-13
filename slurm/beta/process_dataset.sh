#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x.%j.out
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --gres=gpu:4
#SBATCH -w gpuhost012
#SBATCH --cpus-per-task=32
#SBATCH --job-name=process-dataset-world
#SBATCH --mem=400G
##------------------------ End job description ------------------------
module purge && source /gws/nopw/j04/sensecdt/users/mespi/virtualenvs/triffuser/bin/activate


# export PYTHONPATH=/home/users/mespi/projects/triffuser
export PYTHONPATH=$(pwd)
echo "PYTHONPATH: $PYTHONPATH"


# Copy dataset into nvme tmp folder of gpu host. Use rsync in parallel to speed up the process. Wait at the end.
TMP_DIR="/tmp/majorTOM-tmp/world/"
mkdir -p $TMP_DIR

echo "Starting copy at $(date)"
START_TIME=$SECONDS

# As of now, skip copying the dataset as it's already in the tmp folder.
# rsync -az --info=progress2 --partial /gws/nopw/j04/sensecdt/data/internal/majorTOM/world/Core-DEM $TMP_DIR &
# rsync -az --info=progress2 --partial /gws/nopw/j04/sensecdt/data/internal/majorTOM/world/Core-S1RTC $TMP_DIR &
# rsync -az --info=progress2 --partial /gws/nopw/j04/sensecdt/data/internal/majorTOM/world/Core-S2L1C $TMP_DIR &
# rsync -az --info=progress2 --partial /gws/nopw/j04/sensecdt/data/internal/majorTOM/world/Core-S2L2A $TMP_DIR &
# wait

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "Done copying! Total time: $(($ELAPSED_TIME/60)) minutes and $(($ELAPSED_TIME%60)) seconds"

# Encode thumbnails
echo "Starting patchifying images..."
START_TIME=$SECONDS
python3 scripts/prepare_dataset_images.py --subset_path $TMP_DIR --bands thumbnail
# python3 scripts/extract_majortom_feature.py --subset_path $TMP_DIR --bands thumbnail
ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "Done patchifying images! Total time: $(($ELAPSED_TIME/60)) minutes and $(($ELAPSED_TIME%60)) seconds"

# Copy back to scratch
SAVE_DIR="/gws/nopw/j04/sensecdt/data/internal/majorTOM/world/encoded_world_thumbnail/"
mkdir -p $SAVE_DIR
echo "Copying back to $SAVE_DIR"
rsync -az --info=progress2 --partial "$TMP_DIR/encoded_world_thumbnail/train" "$SAVE_DIR" &
rsync -az --info=progress2 --partial "$TMP_DIR/encoded_world_thumbnail/test" "$SAVE_DIR" &
wait

echo "Done copying back to $SAVE_DIR"