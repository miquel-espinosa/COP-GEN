#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x.%j.out
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:4
#SBATCH -x gpuhost015,gpuhost004,gpuhost016
#SBATCH --cpus-per-task=32
#SBATCH --job-name=fast_copy_S1RTC
#SBATCH --mem=400G
##------------------------ End job description ------------------------
module purge && source /gws/nopw/j04/sensecdt/users/mespi/virtualenvs/triffuser/bin/activate

# Copy S1RTC tar.gz data from gws to pw3
mkdir -p /work/scratch-pw3/mespi/majorTOM/world/Core-S1RTC-tar-gz
find /gws/nopw/j04/sensecdt/data/internal/majorTOM/world/Core-S1RTC-tar-gz -maxdepth 1 -name '*.tar.gz' \
  | xargs -P 32 -I{} cp -v {} /work/scratch-pw3/mespi/majorTOM/world/Core-S1RTC-tar-gz/


# Finally copy the .cache file
cp /gws/nopw/j04/sensecdt/data/internal/majorTOM/world/Core-S1RTC-tar-gz/.cache_S1RTC.pkl /work/scratch-pw3/mespi/majorTOM/world/Core-S1RTC-tar-gz/.cache_S1RTC.pkl