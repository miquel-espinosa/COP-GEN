#!/bin/bash

NUM_SAMPLES=4

# Folder with few-images
ROOT_FOLDER=data/majorTOM/loop_subset
INIT_FOLDER=real_init
EXP_FOLDER=start_real_s2l2a

# HELPER IDS
S1_TO_S2_FOLDER_NAME=condition_s1rtc_generate_dem_s2l1c_s2l2a_${NUM_SAMPLES}samples
S2_TO_S1_FOLDER_NAME=condition_s2l2a_generate_dem_s1rtc_s2l1c_${NUM_SAMPLES}samples

# Make folder structure
mkdir -p $ROOT_FOLDER/$INIT_FOLDER/dem
mkdir -p $ROOT_FOLDER/$INIT_FOLDER/s1_rtc
mkdir -p $ROOT_FOLDER/$INIT_FOLDER/s2_l2a
mkdir -p $ROOT_FOLDER/$INIT_FOLDER/s2_l1c

# Copy images
# Define subset of images to copy
IMAGE_IDS=("495U_80R_0" "495U_80R_10" "505U_76R_0" "505U_76R_9")
SRC_PATH="data/majorTOM/northern_italy/northern_italy_thumbnail_png/test"

# Copy each image for each modality
for img_id in "${IMAGE_IDS[@]}"; do
  # Copy DEM images
  cp "$SRC_PATH/DEM_thumbnail/${img_id}.png" "$ROOT_FOLDER/$INIT_FOLDER/dem/"
  
  # Copy S1_RTC images
  cp "$SRC_PATH/S1RTC_thumbnail/${img_id}.png" "$ROOT_FOLDER/$INIT_FOLDER/s1_rtc/"
  
  # Copy S2_L2A images
  cp "$SRC_PATH/S2L2A_thumbnail/${img_id}.png" "$ROOT_FOLDER/$INIT_FOLDER/s2_l2a/"
  
  # Copy S2_L1C images
  cp "$SRC_PATH/S2L1C_thumbnail/${img_id}.png" "$ROOT_FOLDER/$INIT_FOLDER/s2_l1c/"
done

echo "Done copying subset images images"

# ------------------------------------------------------------------------------------------------
# START
# ------------------------------------------------------------------------------------------------

# Define a function to handle the repeated pattern
run_triffuser() {
    local loop_num=$1
    local input_path=$2
    local output_name=$3
    local condition=$4
    local generate=$5
    local desc=$6
    
    echo "Starting. Loop $loop_num: $desc"
    python3 sample_n_triffuser.py \
        --config configs/majortom/discrete/lmdb/world_dems1s2s2_triffuser.py \
        --data_path "$input_path" \
        --data_type folder-img \
        --nnet_path models/dems1s2s2/114000.ckpt/nnet_ema_114000.pth \
        --n_mod 4 \
        --n_samples $NUM_SAMPLES \
        --condition $condition \
        --generate $generate \
        --output_path "$ROOT_FOLDER/$EXP_FOLDER/$output_name" \
        --save_as pngs
    
    echo "Done. Loop $loop_num: $desc"
}

# S2 to S1 - First loop (starting from real data)
#                FROM                                                                              TO
run_triffuser 1 "$ROOT_FOLDER/real_init/s2_l2a"                                                    "v1_real_s2l2a_to_rest" \
                "s2_l2a"                                                                           "dem,s2_l1c,s1_rtc" \
                "Real S2L2A to DEM, S1RTC, S2L1C"

# S1 to S2 - Second loop
#                FROM                                                                              TO
run_triffuser 2 "$ROOT_FOLDER/$EXP_FOLDER/v1_real_s2l2a_to_rest/$S2_TO_S1_FOLDER_NAME/s1_rtc"      "v2_generated_s1rtc_to_rest" \
                "s1_rtc"                                                                           "dem,s2_l1c,s2_l2a" \
                "From generated S1RTC, generate DEM, S2L1C, S2L2A"

exit 0

# S2 to S1 - Third loop
#                FROM                                                                              TO
run_triffuser 3 "$ROOT_FOLDER/$EXP_FOLDER/v2_generated_s1rtc_to_rest/$S1_TO_S2_FOLDER_NAME/s2_l2a" "v3_generated_s2l2a_to_rest" \
                "s2_l2a"                                                                           "dem,s2_l1c,s1_rtc" \
    "From generated S2L2A, generate DEM, S1RTC, S2L1C"

# S1 to S2 - Fourth loop
#                FROM                                                                              TO
run_triffuser 4 "$ROOT_FOLDER/$EXP_FOLDER/v3_generated_s2l2a_to_rest/$S2_TO_S1_FOLDER_NAME/s1_rtc" "v4_generated_s1rtc_to_rest" \
                "s1_rtc"                                                                           "dem,s2_l1c,s2_l2a" \
                "From generated S1RTC, generate DEM, S2L1C, S2L2A"

# S2 to S1 - Fifth loop
#                FROM                                                                              TO
run_triffuser 5 "$ROOT_FOLDER/$EXP_FOLDER/v4_generated_s1rtc_to_rest/$S1_TO_S2_FOLDER_NAME/s2_l2a" "v5_generated_s2l2a_to_rest" \
                "s2_l2a"                                                                           "dem,s2_l1c,s1_rtc" \
                "From generated S2L2A, generate DEM, S1RTC, S2L1C"

# S1 to S2 - Sixth loop
#                FROM                                                                              TO
run_triffuser 6 "$ROOT_FOLDER/$EXP_FOLDER/v5_generated_s2l2a_to_rest/$S2_TO_S1_FOLDER_NAME/s1_rtc" "v6_generated_s1rtc_to_rest" \
                "s1_rtc"                                                                           "dem,s2_l1c,s2_l2a" \
                "From generated S1RTC, generate DEM, S2L1C, S2L2A"

# S2 to S1 - Seventh loop
#                FROM                                                                              TO
run_triffuser 7 "$ROOT_FOLDER/$EXP_FOLDER/v6_generated_s1rtc_to_rest/$S1_TO_S2_FOLDER_NAME/s2_l2a" "v7_generated_s2l2a_to_rest" \
                "s2_l2a"                                                                           "dem,s2_l1c,s1_rtc" \
                "From generated S2L2A, generate DEM, S1RTC, S2L1C"

# S1 to S2 - Eighth loop
#                FROM                                                                              TO
run_triffuser 8 "$ROOT_FOLDER/$EXP_FOLDER/v7_generated_s2l2a_to_rest/$S2_TO_S1_FOLDER_NAME/s1_rtc" "v8_generated_s1rtc_to_rest" \
                "s1_rtc"                                                                           "dem,s2_l1c,s2_l2a" \
                "From generated S1RTC, generate DEM, S2L1C, S2L2A"

# S2 to S1 - Ninth loop
#                FROM                                                                              TO
run_triffuser 9 "$ROOT_FOLDER/$EXP_FOLDER/v8_generated_s1rtc_to_rest/$S1_TO_S2_FOLDER_NAME/s2_l2a" "v9_generated_s2l2a_to_rest" \
                "s2_l2a"                                                                           "dem,s2_l1c,s1_rtc" \
                "From generated S2L2A, generate DEM, S1RTC, S2L1C"

# S1 to S2 - Tenth loop
#                FROM                                                                              TO
run_triffuser 10 "$ROOT_FOLDER/$EXP_FOLDER/v9_generated_s2l2a_to_rest/$S2_TO_S1_FOLDER_NAME/s1_rtc" "v10_generated_s1rtc_to_rest" \
                "s1_rtc"                                                                           "dem,s2_l1c,s2_l2a" \
                "From generated S1RTC, generate DEM, S2L1C, S2L2A"

# S2 to S1 - Eleventh loop
#                FROM                                                                              TO
run_triffuser 11 "$ROOT_FOLDER/$EXP_FOLDER/v10_generated_s1rtc_to_rest/$S1_TO_S2_FOLDER_NAME/s2_l2a" "v11_generated_s2l2a_to_rest" \
                "s2_l2a"                                                                           "dem,s2_l1c,s1_rtc" \
                "From generated S2L2A, generate DEM, S1RTC, S2L1C"

# S1 to S2 - Twelfth loop
#                FROM                                                                              TO
run_triffuser 12 "$ROOT_FOLDER/$EXP_FOLDER/v11_generated_s2l2a_to_rest/$S2_TO_S1_FOLDER_NAME/s1_rtc" "v12_generated_s1rtc_to_rest" \
                "s1_rtc"                                                                           "dem,s2_l1c,s2_l2a" \
                "From generated S1RTC, generate DEM, S2L1C, S2L2A"

# S2 to S1 - Thirteenth loop
#                FROM                                                                              TO
run_triffuser 13 "$ROOT_FOLDER/$EXP_FOLDER/v12_generated_s1rtc_to_rest/$S1_TO_S2_FOLDER_NAME/s2_l2a" "v13_generated_s2l2a_to_rest" \
                "s2_l2a"                                                                           "dem,s2_l1c,s1_rtc" \
                "From generated S2L2A, generate DEM, S1RTC, S2L1C"

# S1 to S2 - Fourteenth loop
#                FROM                                                                              TO
run_triffuser 14 "$ROOT_FOLDER/$EXP_FOLDER/v13_generated_s2l2a_to_rest/$S2_TO_S1_FOLDER_NAME/s1_rtc" "v14_generated_s1rtc_to_rest" \
                "s1_rtc"                                                                           "dem,s2_l1c,s2_l2a" \
                "From generated S1RTC, generate DEM, S2L1C, S2L2A"