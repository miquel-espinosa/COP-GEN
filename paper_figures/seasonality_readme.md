# Seasonality Benchmark Notes

Reference notes for creating and evaluating a compact dataset that targets seasonality behaviors in COPGEN.

## Selected Tiles

- `770U_115R` – Northern Sweden  
- `573U_120R` – Poland  
- `521U_130R` – Central Europe  
- `463U_755L` – North America (Midwest)  
- `519U_865L` – North America (Rockies)  
- `172D_543L` – South America  
- `765U_557L` – Alaska  
- `332U_94R` – Sahara  
- `362D_1427R` – Australia  

## Dataset Preparation

Create a lightweight dataset extracted from `./data/majorTOM/test_dataset_copgen` for the nine focus tiles.

```bash
TILES=("770U_115R" "573U_120R" "521U_130R" "463U_755L" "519U_865L" \
       "172D_543L" "765U_557L" "332U_94R" "362D_1427R")

ROOT="./data/majorTOM/test_dataset_copgen"
OUT="./paper_figures/paper_figures_datasets/one_tile_dataset_seasonality"

for TILE in "${TILES[@]}"; do
    echo "Processing $TILE ..."
    python3 paper_figures/custom_dataset.py \
        --root_dataset "$ROOT" \
        --output_dataset "$OUT/$TILE" \
        --grid_ids "$TILE"
done
```

## Experiment 1 — Lat/Lon Inputs

Run COPGEN with elevation, land use, cloud mask, and lat/lon inputs. Results land in `one_tile_dataset_latlon_seasonality`.

```bash
SEEDS=(12 23 34 45 56 67 78 89 90 101 111 121 131 141 151 161 171 181 191 201 211 221 231 241 251)
MODEL=./models/copgen/cop_gen_base/500000_nnet_ema.pth
CONFIG=./configs/copgen/discrete/cop_gen_base.py
ROOT=./paper_figures/paper_figures_datasets/one_tile_dataset_latlon_seasonality

for TILE in "${TILES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo -e "\033[32mRunning TILE $TILE with SEED $SEED\033[0m"
        python3 copgen_cli.py \
            --dataset-root "$ROOT/$TILE/" \
            --model "$MODEL" \
            --config "$CONFIG" \
            --seed "$SEED" \
            --inputs DEM LULC cloud_mask lat_lon \
            --batch-size 1 \
            --vis-every 1
    done
done
```

## Experiment 2 — Lat/Lon + Timestamp Inputs

Adds timestamp conditioning and writes outputs to `one_tile_dataset_latlon_timestamp_seasonality`.

```bash
ROOT=./paper_figures/paper_figures_datasets/one_tile_dataset_latlon_timestamp_seasonality

for TILE in "${TILES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo -e "\033[32mRunning TILE $TILE with SEED $SEED\033[0m"
        python3 copgen_cli.py \
            --dataset-root "$ROOT/$TILE/" \
            --model "$MODEL" \
            --config "$CONFIG" \
            --seed "$SEED" \
            --inputs DEM LULC cloud_mask lat_lon timestamps \
            --batch-size 1 \
            --vis-every 1
    done
done
```

## Tile Metadata

| Region         | Tile        | Tile Part 1 | Tile Part 2 | Lat / Lon       | Timestamp   | New Timestamp |
| -------------- | ----------- | ----------- | ----------- | --------------- | ----------- | ------------- |
| Australia      | 362D_1427R  | 362D        | 1427R       | -32.47, 152.04  | 29-09-2019  | 01-02-2020    |
| Sweden         | 770U_115R   | 770U        | 115R        | 69.21, 29.16    | 01-04-2018  | 01-07-2014    |
| Poland         | 573U_120R   | 573U        | 120R        | 51.51, 17.37    | 08-07-2023  | 01-01-2022    |
| Europe         | 521U_130R   | 521U        | 130R        | 46.84, 17.12    | 13-06-2019  | 12-11-2020    |
| N. America     | 463U_755L   | 463U        | 755L        | 41.63, -90.60   | 12-12-2020  | 06-06-2022    |
| N. America     | 519U_865L   | 519U        | 865L        | 46.66, -113.05  | 14-09-2023  | 05-01-2024    |
| S. America     | 172D_543L   | 172D        | 543L        | -15.40, -50.56  | 20-05-2021  | 01-12-2022    |
| Alaska         | 765U_557L   | 765U        | 557L        | 68.76, -137.69  | 22-02-2021  | 16-08-2022    |
| Sahara         | 332U_94R    | 332U        | 94R         | 29.87, 9.78     | 20-01-2021  | 20-08-2023    |

## Timestamp Swap Workflow

Use the snippet below to rerun a tile with overridden inputs while preserving the original output directory.

Run script `./paper_figures/scripts/seasonality_complementary.sh` to run the experiments.

```bash
# EUROPE (521U_130R) TILE INFORMATION:
TILE_PART1="521U"
TILE_PART2="130R"
TIMESTAMP="13-06-2019"
LAT_LON="46.84,17.12"
NEW_TIMESTAMP="12-11-2020"

TILE_PATH="paper_figures/paper_figures_datasets/one_tile_dataset_latlon_timestamp_seasonality/${TILE_PART1}_${TILE_PART2}"
mv "$TILE_PATH/outputs/copgen" "$TILE_PATH/outputs/copgen-$TIMESTAMP"

SEEDS=(12 23 34 45 56 67 78 89 90 101 111 121 131 141 151 161 171 181 191 201 211 221 231 241 251)

for SEED in "${SEEDS[@]}"; do
    python3 copgen_cli.py \
        --dataset-root "$TILE_PATH" \
        --model ./models/copgen/cop_gen_base/500000_nnet_ema.pth \
        --config ./configs/copgen/discrete/cop_gen_base.py \
        --seed "$SEED" \
        --vis-every 1 \
        --batch-size 1 \
        --inputs DEM LULC lat_lon timestamps \
        --inputs-paths \
            DEM="$TILE_PATH/Core-DEM/$TILE_PART1/$TILE_PART2/id/DEM.tif" \
            LULC=$TILE_PATH/Core-LULC/$TILE_PART1/$TILE_PART2/*.tif \
            lat_lon="$LAT_LON" \
            timestamps="$NEW_TIMESTAMP"
done
```
