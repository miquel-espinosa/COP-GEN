#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x.%j.out
#SBATCH --account=sensecdt
#SBATCH --qos=standard
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --job-name=copy-to-home
#SBATCH --mem=100G
##------------------------ End job description ------------------------
module purge

# Tar the environment
cd /gws/nopw/j04/sensecdt/users/mespi/virtualenvs/triffuser && tar -czf triffuser.tar.gz .

echo "Environment tarred"

# Copy the environment to the home directory
rsync -avz --info=progress2 --partial /gws/nopw/j04/sensecdt/users/mespi/virtualenvs/triffuser.tar.gz /home/mespi/virtualenvs/

echo "Environment copied to home directory"

# Unpack the environment
cd /home/mespi/virtualenvs/ && tar -xzf triffuser.tar.gz

echo "Environment unpacked"

# Activate the environment
source /home/mespi/virtualenvs/triffuser/bin/activate

echo "Environment activated"