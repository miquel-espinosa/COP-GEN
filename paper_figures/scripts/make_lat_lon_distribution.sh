TILES=("143D_1481R" "95U_112R" "195D_669L" "211D_500R" "215U_1019L" "248U_978R" "250U_409R" "256U_1125L" "272D_1525R" "227D_564L")
# SEEDS=(111 222 333 444 555 666 777 888 12 23 34 45 56 67 78 89 90 13 24 35 46 57 68 79 80)
SEEDS=(101 202 303 404 505 606 707 808 909 1010 1111 1212 1313 1414 1515 1616 1717 1818 1919 2020 2121 2222 2323 2424 2525)

MODEL=./models/copgen/cop_gen_base/500000_nnet_ema.pth
CONFIG=./configs/copgen/discrete/cop_gen_base.py
ROOT=./paper_figures/paper_figures_datasets/one_tile_datasets_DEM_LULC_timestamps_to_lat_lon

for TILE in "${TILES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        
        echo -e "\033[32mRunning TILE $TILE with SEED $SEED\033[0m"

        python3 copgen_cli.py \
            --dataset-root "$ROOT/$TILE/" \
            --model "$MODEL" \
            --config "$CONFIG" \
            --seed "$SEED" \
            --inputs DEM LULC cloud_mask timestamps \
            --batch-size 1 \
            --vis-every 1

    done
done

TILES=("143D_1481R" "95U_112R" "195D_669L" "211D_500R" "215U_1019L" "248U_978R" "250U_409R" "256U_1125L" "272D_1525R" "227D_564L")
# SEEDS=(111 222 333 444 555 666 777 888 12 23 34 45 56 67 78 89 90 13 24 35 46 57 68 79 80)
SEEDS=(101 202 303 404 505 606 707 808 909 1010 1111 1212 1313 1414 1515 1616 1717 1818 1919 2020 2121 2222 2323 2424 2525)

ROOT=./paper_figures/paper_figures_datasets/one_tile_datasets_DEM_LULC_timestamps_to_lat_lon

for TILE in "${TILES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        
        echo -e "\033[31mRunning TILE $TILE with seed $SEED\033[0m"

        python3 terramind_cli.py \
            --dataset_root "$ROOT/$TILE" \
            --input DEM LULC \
            --output coords \
            --seed "$SEED"
    done
done



# Visualise plots

TILES=("143D_1481R" "95U_112R" "195D_669L" "211D_500R" "215U_1019L" "248U_978R" "250U_409R" "256U_1125L" "272D_1525R" "227D_564L")
for TILE in "${TILES[@]}"; do
    python3 paper_figures/lat_lon_comparison.py ./paper_figures/paper_figures_datasets/one_tile_datasets_DEM_LULC_timestamps_to_lat_lon/$TILE 
done