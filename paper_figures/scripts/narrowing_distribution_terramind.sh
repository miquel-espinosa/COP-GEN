# Available modalities: lat_lon, timestamps, DEM, LULC, S1RTC, S2L1C, S2L2A, cloud_mask
# We will evaluate the distribution of the generated outputs by adding more input modalities one by one

# ONLY DEM

TILES=("143D_1481R" "95U_112R" "195D_669L" "211D_500R" "215U_1019L" "248U_978R" "250U_409R" "256U_1125L" "272D_1525R")
SEEDS=(111 222 333 444 555 666 777 888 12 23 34 45 56 67 78 89 90 13 24 35 46 57 68 79 80)
ROOT=./paper_figures/paper_figures_datasets/one_tile_distribution_narrowing

# DEM
for TILE in "${TILES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        
        echo -e "\033[32mRunning TILE $TILE with SEED $SEED\033[0m"
        python terramind_cli.py --input DEM --output S2L2A --dataset_root $ROOT/$TILE --seed $SEED

    done
done

# DEM + LULC
for TILE in "${TILES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        
        echo -e "\033[32mRunning TILE $TILE with SEED $SEED\033[0m"
        python terramind_cli.py --input DEM LULC --output S2L2A --dataset_root $ROOT/$TILE --seed $SEED

    done
done

# DEM + LULC + S1RTC
for TILE in "${TILES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        
        echo -e "\033[32mRunning TILE $TILE with SEED $SEED\033[0m"
        python terramind_cli.py --input DEM LULC S1RTC --output S2L2A --dataset_root $ROOT/$TILE --seed $SEED

    done
done

# DEM + LULC + S1RTC + coords
for TILE in "${TILES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        
        echo -e "\033[32mRunning TILE $TILE with SEED $SEED\033[0m"
        python terramind_cli.py --input DEM LULC S1RTC coords --output S2L2A --dataset_root $ROOT/$TILE --seed $SEED

    done
done

# DEM + LULC + S1RTC + coords + S2L1C
for TILE in "${TILES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        
        echo -e "\033[32mRunning TILE $TILE with SEED $SEED\033[0m"
        python terramind_cli.py --input DEM LULC S1RTC coords S2L1C --output S2L2A --dataset_root $ROOT/$TILE --seed $SEED

    done
done