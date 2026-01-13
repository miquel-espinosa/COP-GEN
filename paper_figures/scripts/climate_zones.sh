#!/bin/bash

# This script will generate lat-lon coordinates given some LULC class.
# We want to see if the generated outputs match with the climate zones.

BASE_TILE="./paper_figures/paper_figures_datasets/one_tile_clean/143D_1481R"
DATASET_ROOT="./paper_figures/paper_figures_datasets/climate_zones_copgen"
FROM_SEED=100
TO_SEED=150
MODEL=./models/copgen/cop_gen_base/500000_nnet_ema.pth
CONFIG=./configs/copgen/discrete/cop_gen_base.py
LULC_CLASSES=("none/nodata" "water" "trees" "flooded vegetation" "crops" "built-up areas" "bare ground" "snow/ice" "clouds" "rangeland")
CLOUD_CLASS="clear"

mkdir -p $DATASET_ROOT

for LULC_CLASS in "${LULC_CLASSES[@]}"; do
    cp -r $BASE_TILE $DATASET_ROOT/$LULC_CLASS
    for SEED in $(seq $FROM_SEED $TO_SEED); do
        echo -e "\033[32mRunning $LULC_CLASS with SEED $SEED\033[0m"
        python3 copgen_cli.py \
            --dataset-root "$DATASET_ROOT/$LULC_CLASS/" \
            --model "$MODEL" \
            --config "$CONFIG" \
            --seed "$SEED" \
            --inputs LULC cloud_mask \
            --inputs-paths $LULC_CLASS $CLOUD_CLASS \
            --batch-size 1 \
            --vis-every 1
    done
done



# TERRAMIND

BASE_TILE="./paper_figures/paper_figures_datasets/one_tile_clean/143D_1481R"
DATASET_ROOT="./paper_figures/paper_figures_datasets/climate_zones_terramind"
FROM_SEED=25
TO_SEED=50
LULC_CLASSES=("water" "trees" "flooded vegetation" "crops" "built area" "bare ground" "snow/ice" "rangeland" "clouds")


for LULC_CLASS in "${LULC_CLASSES[@]}"; do
    cp -r $BASE_TILE $DATASET_ROOT/$LULC_CLASS
    for SEED in $(seq $FROM_SEED $TO_SEED); do
        echo -e "\033[32mRunning $LULC_CLASS with SEED $SEED\033[0m"
        python3 terramind_cli.py \
        --dataset_root "$DATASET_ROOT/$LULC_CLASS/" \
        --input LULC \
        --inputs-paths $LULC_CLASS \
        --output coords \
        --seed $SEED
        python3 terramind_cli.py \
        --dataset_root "$DATASET_ROOT/$LULC_CLASS/" \
        --input LULC \
        --inputs-paths $LULC_CLASS \
        --output S2L2A \
        --seed $SEED
    done
done





# Fixing the snow/ice class

# COPGEN First part

BASE_TILE="./paper_figures/paper_figures_datasets/one_tile_clean/143D_1481R"
DATASET_ROOT="./paper_figures/paper_figures_datasets/climate_zones_copgen"
FROM_SEED=0
TO_SEED=50
MODEL=./models/copgen/cop_gen_base/500000_nnet_ema.pth
CONFIG=./configs/copgen/discrete/cop_gen_base.py
LULC_CLASS="snow/ice"
CLOUD_CLASS="clear"

mkdir -p $DATASET_ROOT

cp -r $BASE_TILE $DATASET_ROOT/snow_ice
for SEED in $(seq $FROM_SEED $TO_SEED); do
    echo -e "\033[32mRunning snow_ice with SEED $SEED\033[0m"
    python3 copgen_cli.py \
        --dataset-root "$DATASET_ROOT/snow_ice/" \
        --model "$MODEL" \
        --config "$CONFIG" \
        --seed "$SEED" \
        --inputs LULC cloud_mask \
        --inputs-paths $LULC_CLASS $CLOUD_CLASS \
        --batch-size 1 \
        --vis-every 1
done

# COPGEN Second part

BASE_TILE="./paper_figures/paper_figures_datasets/one_tile_clean/143D_1481R"
DATASET_ROOT="./paper_figures/paper_figures_datasets/climate_zones_copgen"
FROM_SEED=100
TO_SEED=150
MODEL=./models/copgen/cop_gen_base/500000_nnet_ema.pth
CONFIG=./configs/copgen/discrete/cop_gen_base.py
LULC_CLASS="snow/ice"
CLOUD_CLASS="clear"

mkdir -p $DATASET_ROOT

cp -r $BASE_TILE $DATASET_ROOT/snow_ice
for SEED in $(seq $FROM_SEED $TO_SEED); do
    echo -e "\033[32mRunning snow_ice with SEED $SEED\033[0m"
    python3 copgen_cli.py \
        --dataset-root "$DATASET_ROOT/snow_ice/" \
        --model "$MODEL" \
        --config "$CONFIG" \
        --seed "$SEED" \
        --inputs LULC cloud_mask \
        --inputs-paths $LULC_CLASS $CLOUD_CLASS \
        --batch-size 1 \
        --vis-every 1
done



# TERRAMIND First part

BASE_TILE="./paper_figures/paper_figures_datasets/one_tile_clean/143D_1481R"
DATASET_ROOT="./paper_figures/paper_figures_datasets/climate_zones_terramind"
FROM_SEED=0
TO_SEED=50
LULC_CLASS="snow/ice"

cp -r $BASE_TILE $DATASET_ROOT/snow_ice
for SEED in $(seq $FROM_SEED $TO_SEED); do
    echo -e "\033[32mRunning snow_ice with SEED $SEED\033[0m"
    python3 terramind_cli.py \
    --dataset_root "$DATASET_ROOT/snow_ice/" \
    --input LULC \
    --inputs-paths $LULC_CLASS \
    --output coords \
    --seed $SEED
    python3 terramind_cli.py \
    --dataset_root "$DATASET_ROOT/snow_ice/" \
    --input LULC \
    --inputs-paths $LULC_CLASS \
    --output S2L2A \
    --seed $SEED
done


# TERRAMIND Second part

BASE_TILE="./paper_figures/paper_figures_datasets/one_tile_clean/143D_1481R"
DATASET_ROOT="./paper_figures/paper_figures_datasets/climate_zones_terramind"
FROM_SEED=100
TO_SEED=150
LULC_CLASS="snow/ice"

cp -r $BASE_TILE $DATASET_ROOT/snow_ice
for SEED in $(seq $FROM_SEED $TO_SEED); do
    echo -e "\033[32mRunning snow_ice with SEED $SEED\033[0m"
    python3 terramind_cli.py \
    --dataset_root "$DATASET_ROOT/snow_ice/" \
    --input LULC \
    --inputs-paths $LULC_CLASS \
    --output coords \
    --seed $SEED
    python3 terramind_cli.py \
    --dataset_root "$DATASET_ROOT/snow_ice/" \
    --input LULC \
    --inputs-paths $LULC_CLASS \
    --output S2L2A \
    --seed $SEED
done