#!/usr/bin/env bash
set -euo pipefail

# ==============================
# Common configuration
# ==============================
SEEDS=(12 23 34 45 56 67 78 89 90 101 111 121 131 141 151 161 171 181 191 201 211 221 231 241 251)

MODEL="./models/copgen/cop_gen_base/500000_nnet_ema.pth"
CONFIG="./configs/copgen/discrete/cop_gen_base.py"
DATASET_ROOT_BASE="paper_figures/paper_figures_datasets/one_tile_dataset_latlon_seasonality"

# ==============================
# Tile definitions
# Each entry:
# TILE_PART1 TILE_PART2 LAT_LON_FROM LAT_LON_TO
# ==============================
TILES=(
  "362D 1427R -32.47,152.04 -76.98,49.21"    # AUSTRALIA to Antarctica
  "770U 115R  69.21,29.16 25.75,25.69"       # SWEDEN to Egypt
  "573U 120R  51.51,17.37 -3.48,-65.16"      # POLAND to Amazonas/Brazil
  "521U 130R  46.84,17.12 -27.08,22.42"      # EUROPE to South Africa
  "463U 755L  41.63,-90.60 28.57,-104.13"    # N. AMERICA to Alamos/Mexico
  "519U 865L  46.66,-113.05 75.46,-41.41"    # N. AMERICA to Greenland
  "172D 543L  -15.40,-50.56 -49.39,-73.45"   # S. AMERICA to Patagonia
  "765U 557L  68.76,-137.69 20.28,-4.40"     # ALASKA to Mali/Sahara
  "332U 94R   29.87,9.78 62.66,97.57"        # SAHARA to Russia
)

# ==============================
# Main loop
# ==============================
for TILE in "${TILES[@]}"; do
    read -r TILE_PART1 TILE_PART2 LAT_LON_FROM LAT_LON_TO <<< "$TILE"

    echo "=============================================="
    echo "Processing tile: ${TILE_PART1}_${TILE_PART2}"
    echo "=============================================="

    TILE_PATH="${DATASET_ROOT_BASE}/${TILE_PART1}_${TILE_PART2}"

    # Backup old copgen outputs
    if [ -d "${TILE_PATH}/outputs/copgen" ]; then
        mv "${TILE_PATH}/outputs/copgen" \
           "${TILE_PATH}/outputs/copgen-from"
    fi

    # Find the LULC file (there's only one .tif in the directory)
    LULC_FILE=$(find "${TILE_PATH}/Core-LULC/${TILE_PART1}/${TILE_PART2}" -name "*.tif" -type f | head -n 1)

    if [ -z "$LULC_FILE" ]; then
        echo "ERROR: No LULC .tif file found for ${TILE_PART1}_${TILE_PART2}"
        exit 1
    fi

    # Seed loop
    for SEED in "${SEEDS[@]}"; do
        echo "Running seed ${SEED} for ${TILE_PART1}_${TILE_PART2}"

        python3 copgen_cli.py \
            --dataset-root "${TILE_PATH}" \
            --model "${MODEL}" \
            --config "${CONFIG}" \
            --seed "${SEED}" \
            --vis-every 1 \
            --batch-size 1 \
            --inputs DEM LULC lat_lon \
            --inputs-paths \
                DEM="${TILE_PATH}/Core-DEM/${TILE_PART1}/${TILE_PART1}_${TILE_PART2}/id/DEM.tif" \
                LULC="${LULC_FILE}" \
                lat_lon="${LAT_LON_TO}"
    done
done
