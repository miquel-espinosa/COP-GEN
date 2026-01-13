#!/bin/bash

export SEN1FLOODS11_ROOT=$(pwd)/data/sen1floods11_v1.1/v1.1
export EXP_GT_S2_GEN_S1_ROOT=$SEN1FLOODS11_ROOT/outputs/GEN_S1_FROM_S2L2A
export SEN1FLOODS11_DATA_192_ROOT=$SEN1FLOODS11_ROOT/data_192/v1.1
export SEN1FLOODS11_DATA_192_DATA_ROOT=$SEN1FLOODS11_DATA_192_ROOT/data/flood_events/HandLabeled

for split in train test val; do
    python3 copgen_eval/sen1floods11/generate_sen1floods11.py \
        --model_path models/copgen/world_12_modalities_large/300000.ckpt/nnet_ema.pth \
        --model_config configs/copgen/discrete/cop_gen_large.py \
        --data_config configs/copgen_eval/sen1floods11/copgen_sen1floods11.yaml \
        --condition_modalities S2L2A_B01_B09 S2L2A_B02_B03_B04_B08 S2L2A_B05_B06_B07_B11_B12_B8A \
        --root_output_path $EXP_GT_S2_GEN_S1_ROOT \
        --split $split \
        --batch_size 16 \
        --visualise_every_n_batches 10
done

# Link 192x192 GT Labels and GT S2 to experiment output folder
ln -sf $SEN1FLOODS11_DATA_192_DATA_ROOT/LabelHand $EXP_GT_S2_GEN_S1_ROOT/v1.1/data/flood_events/HandLabeled/LabelHand

# Link 192x192 GT S2 to experiment output folder
ln -sf $SEN1FLOODS11_DATA_192_DATA_ROOT/S2Hand $EXP_GT_S2_GEN_S1_ROOT/v1.1/data/flood_events/HandLabeled/S2Hand

# Copy generated S1 to correct naming convention
mkdir -p $EXP_GT_S2_GEN_S1_ROOT/v1.1/data/flood_events/HandLabeled/S1Hand
for tif in $EXP_GT_S2_GEN_S1_ROOT/v1.1/data/flood_events/HandLabeled/S1RTC_vh_vv/*.tif; do
    filename=$(basename "$tif")
    new_filename="${filename/S1RTC_vh_vv/S1Hand}"
    cp "$tif" "$EXP_GT_S2_GEN_S1_ROOT/v1.1/data/flood_events/HandLabeled/S1Hand/$new_filename"
done