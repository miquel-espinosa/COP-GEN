#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x.%j.out
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH -x gpuhost015
#SBATCH --cpus-per-task=4
#SBATCH --job-name=copy-s1rtc
#SBATCH --mem=100G
##------------------------ End job description ------------------------
####SBATCH --partition=standard
####SBATCH --account=sensecdt
####SBATCH --qos=standard

module purge && source /gws/nopw/j04/sensecdt/users/mespi/virtualenvs/triffuser/bin/activate

rsync -az --info=progress2 --no-compress /work/scratch-pw2/mespi/majorTOM/world/Core-S1RTC /gws/nopw/j04/sensecdt/data/internal/majorTOM/world