#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x.%j.out
#SBATCH --partition=standard
#SBATCH --account=sensecdt
#SBATCH --qos=high
#SBATCH -x gpuhost015,gpuhost004,gpuhost016
#SBATCH --cpus-per-task=32
#SBATCH --job-name=fast_copy_S2L1C
#SBATCH --mem=400G
##------------------------ End job description ------------------------
module purge && source /gws/nopw/j04/sensecdt/users/mespi/virtualenvs/triffuser/bin/activate

# Copy S2L1C tar.gz data from gws to pw3
mkdir -p /work/scratch-pw2/mespi/majorTOM/world/Core-S2L1C-tar-gz
find /work/scratch-pw3/mespi/majorTOM/world/Core-S2L1C-tar-gz/ -maxdepth 1 -name '*.tar.gz' \
  | xargs -P 32 -I{} cp -v {} /work/scratch-pw2/mespi/majorTOM/world/Core-S2L1C-tar-gz/

# Finally copy the .cache file
cp /work/scratch-pw3/mespi/majorTOM/world/Core-S2L1C-tar-gz/.cache_S2L1C.pkl /work/scratch-pw2/mespi/majorTOM/world/Core-S2L1C-tar-gz/.cache_S2L1C.pkl
