# Available modalities: lat_lon, timestamps, DEM, LULC, S1RTC, S2L1C, S2L2A, cloud_mask
# We will evaluate the distribution of the generated outputs by adding more input modalities one by one

# ONLY DEM and cloud_mask

TILES=("143D_1481R" "95U_112R" "195D_669L" "211D_500R" "215U_1019L" "248U_978R" "250U_409R" "256U_1125L" "272D_1525R")
SEEDS=(111 222 333 444 555 666 777 888 12 23 34 45 56 67 78 89 90 13 24 35 46 57 68 79 80)

MODEL=./models/copgen/cop_gen_base/500000_nnet_ema.pth
CONFIG=./configs/copgen/discrete/cop_gen_base.py
ROOT=./paper_figures/paper_figures_datasets/one_tile_distribution_narrowing

for TILE in "${TILES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        
        echo -e "\033[32mRunning TILE $TILE with SEED $SEED\033[0m"

        python3 copgen_cli.py \
            --dataset-root "$ROOT/$TILE/" \
            --model "$MODEL" \
            --config "$CONFIG" \
            --seed "$SEED" \
            --inputs DEM cloud_mask \
            --batch-size 1 \
            --vis-every 1

    done
done

# DEM + cloud_mask + LULC

TILES=("143D_1481R" "95U_112R" "195D_669L" "211D_500R" "215U_1019L" "248U_978R" "250U_409R" "256U_1125L" "272D_1525R")
SEEDS=(111 222 333 444 555 666 777 888 12 23 34 45 56 67 78 89 90 13 24 35 46 57 68 79 80)

MODEL=./models/copgen/cop_gen_base/500000_nnet_ema.pth
CONFIG=./configs/copgen/discrete/cop_gen_base.py
ROOT=./paper_figures/paper_figures_datasets/one_tile_distribution_narrowing

for TILE in "${TILES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        
        echo -e "\033[32mRunning TILE $TILE with SEED $SEED\033[0m"

        python3 copgen_cli.py \
            --dataset-root "$ROOT/$TILE/" \
            --model "$MODEL" \
            --config "$CONFIG" \
            --seed "$SEED" \
            --inputs DEM cloud_mask LULC \
            --batch-size 1 \
            --vis-every 1

    done
done

# DEM + cloud_mask + LULC + S1RTC

TILES=("143D_1481R" "95U_112R" "195D_669L" "211D_500R" "215U_1019L" "248U_978R" "250U_409R" "256U_1125L" "272D_1525R")
SEEDS=(111 222 333 444 555 666 777 888 12 23 34 45 56 67 78 89 90 13 24 35 46 57 68 79 80)

MODEL=./models/copgen/cop_gen_base/500000_nnet_ema.pth
CONFIG=./configs/copgen/discrete/cop_gen_base.py
ROOT=./paper_figures/paper_figures_datasets/one_tile_distribution_narrowing

for TILE in "${TILES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        
        echo -e "\033[32mRunning TILE $TILE with SEED $SEED\033[0m"

        python3 copgen_cli.py \
            --dataset-root "$ROOT/$TILE/" \
            --model "$MODEL" \
            --config "$CONFIG" \
            --seed "$SEED" \
            --inputs DEM cloud_mask LULC S1RTC \
            --batch-size 1 \
            --vis-every 1

    done
done


# DEM + cloud_mask + LULC + S1RTC + timestamps

TILES=("143D_1481R" "95U_112R" "195D_669L" "211D_500R" "215U_1019L" "248U_978R" "250U_409R" "256U_1125L" "272D_1525R")
SEEDS=(111 222 333 444 555 666 777 888 12 23 34 45 56 67 78 89 90 13 24 35 46 57 68 79 80)

MODEL=./models/copgen/cop_gen_base/500000_nnet_ema.pth
CONFIG=./configs/copgen/discrete/cop_gen_base.py
ROOT=./paper_figures/paper_figures_datasets/one_tile_distribution_narrowing

for TILE in "${TILES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        
        echo -e "\033[32mRunning TILE $TILE with SEED $SEED\033[0m"

        python3 copgen_cli.py \
            --dataset-root "$ROOT/$TILE/" \
            --model "$MODEL" \
            --config "$CONFIG" \
            --seed "$SEED" \
            --inputs DEM cloud_mask LULC S1RTC timestamps \
            --batch-size 1 \
            --vis-every 1

    done
done

# DEM + cloud_mask + LULC + S1RTC + timestamps + lat_lon

TILES=("143D_1481R" "95U_112R" "195D_669L" "211D_500R" "215U_1019L" "248U_978R" "250U_409R" "256U_1125L" "272D_1525R")
SEEDS=(111 222 333 444 555 666 777 888 12 23 34 45 56 67 78 89 90 13 24 35 46 57 68 79 80)

MODEL=./models/copgen/cop_gen_base/500000_nnet_ema.pth
CONFIG=./configs/copgen/discrete/cop_gen_base.py
ROOT=./paper_figures/paper_figures_datasets/one_tile_distribution_narrowing

for TILE in "${TILES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        
        echo -e "\033[32mRunning TILE $TILE with SEED $SEED\033[0m"

        python3 copgen_cli.py \
            --dataset-root "$ROOT/$TILE/" \
            --model "$MODEL" \
            --config "$CONFIG" \
            --seed "$SEED" \
            --inputs DEM cloud_mask LULC S1RTC timestamps lat_lon \
            --batch-size 1 \
            --vis-every 1

    done
done


# DEM + cloud_mask + LULC + S1RTC + timestamps + lat_lon + S2L1C

TILES=("143D_1481R" "95U_112R" "195D_669L" "211D_500R" "215U_1019L" "248U_978R" "250U_409R" "256U_1125L" "272D_1525R")
SEEDS=(111 222 333 444 555 666 777 888 12 23 34 45 56 67 78 89 90 13 24 35 46 57 68 79 80)

MODEL=./models/copgen/cop_gen_base/500000_nnet_ema.pth
CONFIG=./configs/copgen/discrete/cop_gen_base.py
ROOT=./paper_figures/paper_figures_datasets/one_tile_distribution_narrowing

for TILE in "${TILES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        
        echo -e "\033[32mRunning TILE $TILE with SEED $SEED\033[0m"

        python3 copgen_cli.py \
            --dataset-root "$ROOT/$TILE/" \
            --model "$MODEL" \
            --config "$CONFIG" \
            --seed "$SEED" \
            --inputs DEM cloud_mask LULC S1RTC timestamps lat_lon S2L1C \
            --batch-size 1 \
            --vis-every 1

    done
done