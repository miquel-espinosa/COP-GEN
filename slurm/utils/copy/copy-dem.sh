#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x.%j.out
#SBATCH --partition=standard
#SBATCH --account=sensecdt
#SBATCH --qos=standard
#SBATCH --cpus-per-task=1
#SBATCH --job-name=copy-dem
#SBATCH --mem=100G
##------------------------ End job description ------------------------

module purge && source /gws/nopw/j04/sensecdt/users/mespi/virtualenvs/triffuser/bin/activate

rsync -az --info=progress2 --no-compress /work/scratch-pw2/mespi/majorTOM/world/Core-DEM /gws/nopw/j04/sensecdt/data/internal/majorTOM/world