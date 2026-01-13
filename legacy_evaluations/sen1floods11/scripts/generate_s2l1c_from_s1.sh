#!/bin/bash

export SEN1FLOODS11_ROOT=$(pwd)/data/sen1floods11_v1.1/v1.1
export EXP_GT_S1_GEN_S2_ROOT=$SEN1FLOODS11_ROOT/outputs/GEN_S2L1C_FROM_S1
export SEN1FLOODS11_DATA_192_ROOT=$SEN1FLOODS11_ROOT/data_192/v1.1
export SEN1FLOODS11_DATA_192_DATA_ROOT=$SEN1FLOODS11_DATA_192_ROOT/data/flood_events/HandLabeled

for split in train test val; do
    python3 copgen_eval/sen1floods11/generate_sen1floods11.py \
        --model_path models/copgen/world_12_modalities_large/300000.ckpt/nnet_ema.pth \
        --model_config configs/copgen/discrete/cop_gen_large.py \
        --data_config configs/copgen_eval/sen1floods11/copgen_sen1floods11.yaml \
        --condition_modalities S1RTC_vh_vv \
        --root_output_path $EXP_GT_S1_GEN_S2_ROOT \
        --split $split \
        --batch_size 16 \
        --visualise_every_n_batches 10
done

# Link 192x192 GT Labels and GT S1 to experiment output folder
ln -sf $SEN1FLOODS11_DATA_192_DATA_ROOT/S1Hand $EXP_GT_S1_GEN_S2_ROOT/v1.1/data/flood_events/HandLabeled/S1Hand
ln -sf $SEN1FLOODS11_DATA_192_DATA_ROOT/LabelHand $EXP_GT_S1_GEN_S2_ROOT/v1.1/data/flood_events/HandLabeled/LabelHand

# Copy generated S2 to correct naming convention
mkdir -p $EXP_GT_S1_GEN_S2_ROOT/v1.1/data/flood_events/HandLabeled/S2Hand
for tif in $EXP_GT_S1_GEN_S2_ROOT/v1.1/data/flood_events/HandLabeled/S2L1C/*.tif; do
    filename=$(basename "$tif")
    new_filename="${filename/S2L1C/S2Hand}"
    cp "$tif" "$EXP_GT_S1_GEN_S2_ROOT/v1.1/data/flood_events/HandLabeled/S2Hand/$new_filename"
done