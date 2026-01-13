#!/bin/bash

# PATHS
DATA_DIR="./data/majorTOM"
# DATES
START_DATE="2017-01-01"
END_DATE="2025-01-01"
# SOURCES: all the modalities that we want to pair-download.
SOURCES=("Core-S2L2A" "Core-S2L1C" "Core-S1RTC" "Core-DEM" "Core-LULC")
# MODALITIES: the modalities that we actually want to process.
MODALITIES=("Core-S2L2A" "Core-S2L1C" "Core-S1RTC" "Core-DEM" "Core-LULC")
# MODE: download or extract or full.
MODE="full"

# Download region
python3 majortom/download_world.py \
    --data-dir $DATA_DIR \
    --sources "${SOURCES[@]}" \
    --start-date $START_DATE \
    --end-date $END_DATE \
    --cloud-cover 0 10 \
    --subset-name "edinburgh" \
    --bbox -5.0 55.5 -2.3 56.9 \
    --criteria "latest" \
    --modalities "${MODALITIES[@]}" \
    --mode $MODE \
    --revalidate \
    --download-workers 16 \
    --seed 42
