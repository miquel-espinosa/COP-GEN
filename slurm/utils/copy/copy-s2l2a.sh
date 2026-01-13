#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x.%j.out
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH -x gpuhost015
#SBATCH --cpus-per-task=1
#SBATCH --job-name=copy-s2l2a
#SBATCH --mem=100G
##------------------------ End job description ------------------------

module purge && source /gws/nopw/j04/sensecdt/users/mespi/virtualenvs/triffuser/bin/activate

rsync -az --info=progress2 --no-compress /work/scratch-pw2/mespi/majorTOM/world/Core-S2L2A /work/scratch-pw3/mespi/majorTOM/world/Core-S2L2A