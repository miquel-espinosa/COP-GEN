#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x.%j.out
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH -x gpuhost015
#SBATCH --cpus-per-task=16
#SBATCH --job-name=tar-world-S1RTC
#SBATCH --mem=100G
##------------------------ End job description ------------------------
######SBATCH --partition=standard
######SBATCH --account=sensecdt
######SBATCH --qos=standard

module purge && source /gws/nopw/j04/sensecdt/users/mespi/virtualenvs/triffuser/bin/activate

# PATHS
SOURCE_DIR="/work/scratch-pw2/mespi/majorTOM/world/Core-S1RTC"
TARGET_DIR="/gws/nopw/j04/sensecdt/data/internal/majorTOM/world/Core-S1RTC-tar-gz"
NUM_WORKERS=16

python3 scripts/tar_subfolders.py $SOURCE_DIR $TARGET_DIR --workers $NUM_WORKERS