#! /bin/bash

# This script is to generate the spectral profiles for the one tile dataset, for the figures used in the paper

python paper_figures/spectral_profiles.py \
    --tile-root ./paper_figures/paper_figures_datasets/one_tile_datasets_DEM_LULC_to_S2L2A/143D_1481R \
    --terramind-exp input_DEM_LULC_output_S2L2A_seed_111 \
    --copgen-exp input_DEM_LULC_cloud_mask_output_S1RTC_S2L1C_S2L2A_lat_lon_timestamps_seed_111 \
    --seed 42

python paper_figures/spectral_profiles.py \
    --tile-root ./paper_figures/paper_figures_datasets/one_tile_datasets_DEM_LULC_to_S2L2A/195D_669L \
    --terramind-exp input_DEM_LULC_output_S2L2A_seed_34 \
    --copgen-exp input_DEM_LULC_cloud_mask_output_S1RTC_S2L1C_S2L2A_lat_lon_timestamps_seed_34 \
    --seed 42

python paper_figures/spectral_profiles.py \
    --tile-root ./paper_figures/paper_figures_datasets/one_tile_datasets_DEM_LULC_to_S2L2A/211D_500R \
    --terramind-exp input_DEM_LULC_output_S2L2A_seed_78 \
    --copgen-exp input_DEM_LULC_cloud_mask_output_S1RTC_S2L1C_S2L2A_lat_lon_timestamps_seed_78 \
    --seed 42

python paper_figures/spectral_profiles.py \
    --tile-root ./paper_figures/paper_figures_datasets/one_tile_datasets_DEM_LULC_to_S2L2A/215U_1019L \
    --terramind-exp input_DEM_LULC_output_S2L2A_seed_89 \
    --copgen-exp input_DEM_LULC_cloud_mask_output_S1RTC_S2L1C_S2L2A_lat_lon_timestamps_seed_89 \
    --seed 42

python paper_figures/spectral_profiles.py \
    --tile-root ./paper_figures/paper_figures_datasets/one_tile_datasets_DEM_LULC_to_S2L2A/248U_978R \
    --terramind-exp input_DEM_LULC_output_S2L2A_seed_78 \
    --copgen-exp input_DEM_LULC_cloud_mask_output_S1RTC_S2L1C_S2L2A_lat_lon_timestamps_seed_78 \
    --seed 42

python paper_figures/spectral_profiles.py \
    --tile-root ./paper_figures/paper_figures_datasets/one_tile_datasets_DEM_LULC_to_S2L2A/256U_1125L \
    --terramind-exp input_DEM_LULC_output_S2L2A_seed_67 \
    --copgen-exp input_DEM_LULC_cloud_mask_output_S1RTC_S2L1C_S2L2A_lat_lon_timestamps_seed_67 \
    --seed 42

python paper_figures/spectral_profiles.py \
    --tile-root ./paper_figures/paper_figures_datasets/one_tile_datasets_DEM_LULC_to_S2L2A/272D_1525R \
    --terramind-exp input_DEM_LULC_output_S2L2A_seed_45 \
    --copgen-exp input_DEM_LULC_cloud_mask_output_S1RTC_S2L1C_S2L2A_lat_lon_timestamps_seed_45 \
    --seed 42



