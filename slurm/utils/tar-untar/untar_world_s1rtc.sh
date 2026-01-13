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
#SBATCH --job-name=untar-world-S1RTC
#SBATCH --mem=100G
##------------------------ End job description ------------------------
######SBATCH --partition=standard
######SBATCH --account=sensecdt
######SBATCH --qos=standard

module purge && source /gws/nopw/j04/sensecdt/users/mespi/virtualenvs/triffuser/bin/activate

SOURCE_DIR="/gws/nopw/j04/sensecdt/data/internal/majorTOM/world/Core-S1RTC-tar-gz"
TARGET_DIR="/work/scratch-pw3/mespi/majorTOM/world/Core-S1RTC"
TMP_DIR="/work/scratch-pw3/mespi/majorTOM/tmp-s1rtc"

# Create tmp directory if it doesn't exist
mkdir -p $TMP_DIR

./slurm/copy_and_extract.sh $SOURCE_DIR $TARGET_DIR $TMP_DIR

# Remove tmp directory
rm -rf $TMP_DIR
echo "✅ Removed tmp directory"