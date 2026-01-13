#!/usr/bin/env bash
set -euo pipefail

# ==============================
# Common configuration
# ==============================
SEEDS=(12 23 34 45 56 67 78 89 90 101 111 121 131 141 151 161 171 181 191 201 211 221 231 241 251)

MODEL="./models/copgen/cop_gen_base/500000_nnet_ema.pth"
CONFIG="./configs/copgen/discrete/cop_gen_base.py"
DATASET_ROOT_BASE="paper_figures/paper_figures_datasets/one_tile_dataset_latlon_timestamp_seasonality"

# ==============================
# Tile definitions
# Each entry:
# TILE_PART1 TILE_PART2 TIMESTAMP LAT_LON NEW_TIMESTAMP
# ==============================
TILES=(
  "362D 1427R 29-09-2019 -32.47,152.04 01-02-2020"   # AUSTRALIA
  "770U 115R 01-04-2018 69.21,29.16 01-07-2014"     # SWEDEN
  "573U 120R 08-07-2023 51.51,17.37 01-01-2022"     # POLAND
  "521U 130R 13-06-2019 46.84,17.12 12-11-2020"     # EUROPE
  "463U 755L 12-12-2020 41.63,-90.60 06-06-2022"    # N. AMERICA
  "519U 865L 14-09-2023 46.66,-113.05 05-01-2024"   # N. AMERICA
  "172D 543L 20-05-2021 -15.40,-50.56 01-12-2022"   # S. AMERICA
  "765U 557L 22-02-2021 68.76,-137.69 16-08-2022"   # ALASKA
  "332U 94R  20-01-2021 29.87,9.78 20-08-2023"      # SAHARA
)

# ==============================
# Main loop
# ==============================
for TILE in "${TILES[@]}"; do
    read -r TILE_PART1 TILE_PART2 TIMESTAMP LAT_LON NEW_TIMESTAMP <<< "$TILE"

    echo "=============================================="
    echo "Processing tile: ${TILE_PART1}_${TILE_PART2}"
    echo "=============================================="

    TILE_PATH="${DATASET_ROOT_BASE}/${TILE_PART1}_${TILE_PART2}"

    # Backup old copgen outputs
    if [ -d "${TILE_PATH}/outputs/copgen" ]; then
        mv "${TILE_PATH}/outputs/copgen" \
           "${TILE_PATH}/outputs/copgen-${TIMESTAMP}"
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
            --inputs DEM LULC lat_lon timestamps \
            --inputs-paths \
                DEM="${TILE_PATH}/Core-DEM/${TILE_PART1}/${TILE_PART1}_${TILE_PART2}/id/DEM.tif" \
                LULC="${LULC_FILE}" \
                lat_lon="${LAT_LON}" \
                timestamps="${NEW_TIMESTAMP}"
    done
done
