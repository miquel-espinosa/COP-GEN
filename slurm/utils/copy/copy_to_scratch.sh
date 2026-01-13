#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x.%j.out
#SBATCH --partition=par-single
#SBATCH --cpus-per-task=16
#SBATCH --job-name=copy-to-scratch
#SBATCH --mem=100G
##------------------------ End job description ------------------------
module purge && source /gws/nopw/j04/sensecdt/users/mespi/virtualenvs/triffuser/bin/activate

export PYTHONPATH=$(pwd)
echo "PYTHONPATH: $PYTHONPATH"

# Copy dataset into nvme tmp folder of gpu host. Use rsync in parallel to speed up the process. Wait at the end.
TMP_DIR="/work/scratch-nopw2/mespi/majorTOM-clone/world/"
mkdir -p $TMP_DIR

echo "Starting copy at $(date)"
START_TIME=$SECONDS

rsync -az --info=progress2 --partial /gws/nopw/j04/sensecdt/data/internal/majorTOM/world/Core-DEM $TMP_DIR &
rsync -az --info=progress2 --partial /gws/nopw/j04/sensecdt/data/internal/majorTOM/world/Core-S1RTC $TMP_DIR &
rsync -az --info=progress2 --partial /gws/nopw/j04/sensecdt/data/internal/majorTOM/world/Core-S2L1C $TMP_DIR &
rsync -az --info=progress2 --partial /gws/nopw/j04/sensecdt/data/internal/majorTOM/world/Core-S2L2A $TMP_DIR &
wait

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "Done copying! Total time: $(($ELAPSED_TIME/60)) minutes and $(($ELAPSED_TIME%60)) seconds"