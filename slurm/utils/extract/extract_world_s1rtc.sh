#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x.%j.out
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH -x gpuhost015
#SBATCH --cpus-per-task=15
#SBATCH --job-name=extract-world-S1RTC
#SBATCH --mem=100G
##------------------------ End job description ------------------------
######SBATCH --partition=standard
######SBATCH --account=sensecdt
######SBATCH --qos=standard

module purge && source /gws/nopw/j04/sensecdt/users/mespi/virtualenvs/triffuser/bin/activate

# PATHS
DATA_DIR="./scratch-pw2/majorTOM"
# DATES
START_DATE="2017-01-01"
END_DATE="2025-01-01"
# SOURCES: all the modalities that we want to pair-download.
SOURCES=("Core-S2L2A" "Core-S2L1C" "Core-S1RTC" "Core-DEM")
# MODALITIES: the modalities that we actually want to process.
MODALITIES=("Core-S1RTC")
# MODE: download or extract or full.
MODE="extract"

# Download region
python3 majortom/download_world.py \
    --data-dir $DATA_DIR \
    --sources "${SOURCES[@]}" \
    --start-date $START_DATE \
    --end-date $END_DATE \
    --cloud-cover 0 10 \
    --subset-name "world" \
    --bbox -180.0 -90.0 180.0 90.0 \
    --criteria "latest" \
    --modalities "${MODALITIES[@]}" \
    --mode $MODE \
    --revalidate \
    --download-workers 16 \
    --seed 42



# Command for copying source to target directory with rsync in parallel
# NOTE: As of now, this is not working.
# NUM_THREADS=32
# ~/msrsync -p $NUM_THREADS --rsync "-az --info=progress2 --partial" /source/ /target/
