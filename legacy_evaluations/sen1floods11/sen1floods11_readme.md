### All Available Modalities:
```
  - 3d_cartesian_lat_lon
  - DEM_DEM
  - LULC_LULC
  - S1RTC_vh_vv
  - S2L1C_B01_B09_B10
  - S2L1C_B02_B03_B04_B08
  - S2L1C_B05_B06_B07_B11_B12_B8A
  - S2L1C_cloud_mask
  - S2L2A_B01_B09
  - S2L2A_B02_B03_B04_B08
  - S2L2A_B05_B06_B07_B11_B12_B8A
  - mean_timestamps
```

### Known issues:

!!! The preprocessing for S1 coming from Sen1Floods11 dataset needs to be fixed so that it matches the input distribution that our cop-gen diffusion model is expecting.

This needs to be fixed by playing with the preprocessing in `ddm/pre_post_process_data.py `. For example, by skipping the log10 transformation (as sen1floods11 dataset is already in dB scale). Also by playing the min max scaling, etc.

Basically, needs debugging, I haven't had time to explore it yet.

## Steps

## 1. Dataset setup and model checkpoints

Place sen1floods11 dataset in the data directory.

```
./data/sen1floods11_v1.1/v1.1/
```

Download autoencoder checkpoints from https://huggingface.co/mespinosami/cop-gen-encoders and place them in

```
Loading autoencoder from models/vae/DEM_64x64_DEM_latent_8/model-50-ema.pth
Loading autoencoder from models/vae/LULC_192x192_LULC_latent_8/model-50-ema.pth
Loading autoencoder from models/vae/S1RTC_192x192_VV_VH_latent_8/model-50-ema.pth
Loading autoencoder from models/vae/S2L1C_32x32_B1_9_10_latent_8/model-50-ema.pth
Loading autoencoder from models/vae/S2L1C_192x192_B4_3_2_8_latent_8/model-50-ema.pth
Loading autoencoder from models/vae/S2L1C_96x96_B5_6_7_8A_11_12_latent_8/model-50-ema.pth
Loading autoencoder from models/vae/S2L1C_192x192_cloud_mask_latent_8/model-50-ema.pth
Loading autoencoder from models/vae/S2L2A_32x32_B1_9_latent_8/model-50-ema.pth
Loading autoencoder from models/vae/S2L2A_192x192_B4_3_2_8_latent_8/model-50-ema.pth
Loading autoencoder from models/vae/S2L2A_96x96_B5_6_7_8A_11_12_latent_8/model-50-ema.pth
```

Download model checkpoint from https://huggingface.co/mespinosami/cop-gen-large/tree/main
```
models/copgen/world_12_modalities_large/300000.ckpt/nnet_ema.pth
```

### Configuration files

We have multiple configuration files for the different experiments.


### TLDR

Make crops of 192x192 for the GT data.

```bash 
# Generate crops
./legacy_evaluations/sen1floods11/scripts/crop_192.sh
```

Generate data

```bash
# Generate S1 from S2
./legacy_evaluations/sen1floods11/scripts/generate_s1_from_s2.sh

# Generate S1 from S2L1C
./legacy_evaluations/sen1floods11/scripts/generate_s1_from_s2l1c.sh

# Generate S1 from S2L2A
./legacy_evaluations/sen1floods11/scripts/generate_s1_from_s2l2a.sh

# Generate S2L1C from S1
./legacy_evaluations/sen1floods11/scripts/generate_s2l1c_from_s1.sh

# Generate S2L2A from S1
./legacy_evaluations/sen1floods11/scripts/generate_s2l2a_from_s1.sh
```

Train models on GT and generated data

```bash
# Train model on GT S1 and GT S2 data
./legacy_evaluations/sen1floods11/scripts/train_gt.sh

# Train model on GENERATED S1 from S2 L1C+L2A combined GT data
./legacy_evaluations/sen1floods11/scripts/train_gen_s1_from_s2.sh

# Train model on GENERATED S1 from S2 L1C GT data
./legacy_evaluations/sen1floods11/scripts/train_gen_s1_from_s2l1c.sh

# Train model on GENERATED S1 from S2 L2A GT data
./legacy_evaluations/sen1floods11/scripts/train_gen_s1_from_s2l2a.sh

# Train model on GENERATED S2L1C from S1 GT data
./legacy_evaluations/sen1floods11/scripts/train_gen_s2l1c_from_s1.sh

# Train model on GENERATED S2L2A from S1 GT data
./legacy_evaluations/sen1floods11/scripts/train_gen_s2l2a_from_s1.sh
```

### Create data_192 (center-cropped)

Create a dataset with 192x192 center cropped images for S1Hand, S2Hand, and LabelHand.

Run script:
```bash
# Define paths
export SEN1FLOODS11_ROOT=$(pwd)/data/sen1floods11_v1.1/v1.1
export SEN1FLOODS11_DATA_192_ROOT=$SEN1FLOODS11_ROOT/data_192/v1.1
export SEN1FLOODS11_DATA_192_DATA_ROOT=$SEN1FLOODS11_DATA_192_ROOT/data/flood_events/HandLabeled
# Copy splits and metadata
mkdir -p $SEN1FLOODS11_DATA_192_DATA_ROOT
cp -r $SEN1FLOODS11_ROOT/splits $SEN1FLOODS11_DATA_192_ROOT/splits
cp $SEN1FLOODS11_ROOT/Sen1Floods11_Metadata.geojson $SEN1FLOODS11_DATA_192_ROOT/Sen1Floods11_Metadata.geojson
# Center-crop
python3 legacy_evaluations/sen1floods11/center_crop_sen1floods11.py --size 192 \
  --src-root $SEN1FLOODS11_ROOT/data/flood_events/HandLabeled \
  --dst-root $SEN1FLOODS11_DATA_192_DATA_ROOT
```

Output will be stored in `$SEN1FLOODS11_DATA_192_ROOT`.


### Generate S1 from S2

Generate S2 from S1:

```bash

export EXP_GT_S1_GEN_S2_ROOT=$SEN1FLOODS11_ROOT/outputs/GT_S1_GEN_S2

python3 legacy_evaluations/sen1floods11/generate_sen1floods11.py \
    --model_path models/copgen/world_12_modalities_large/300000.ckpt/nnet_ema.pth \
    --model_config configs/copgen/discrete/cop_gen_large.py \
    --data_config configs/legacy_evaluations/sen1floods11/copgen_sen1floods11.yaml \
    --condition_modalities S1RTC_vh_vv \
    --root_output_path $EXP_GT_S1_GEN_S2_ROOT \
    --split train \
    --batch_size 16 \
    --visualise_every_n_batches 10
```

Make the necessary symbolic links to the ground truth labels and S1, and change the file naming convention of the generated S2.

```bash
# Link 192x192 GT Labels and GT S1 to experiment output folder
ln -s $SEN1FLOODS11_DATA_192_DATA_ROOT/S1Hand $EXP_GT_S1_GEN_S2_ROOT/v1.1/data/flood_events/HandLabeled/S1Hand
ln -s $SEN1FLOODS11_DATA_192_DATA_ROOT/LabelHand $EXP_GT_S1_GEN_S2_ROOT/v1.1/data/flood_events/HandLabeled/LabelHand

# Copy generated S2 to correct naming convention
mkdir -p $EXP_GT_S1_GEN_S2_ROOT/v1.1/data/flood_events/HandLabeled/S2Hand
for tif in $EXP_GT_S1_GEN_S2_ROOT/v1.1/data/flood_events/HandLabeled/S2L1C/*.tif; do
    filename=$(basename "$tif")
    new_filename="${filename/S2L2A/S2Hand}"
    cp "$tif" "$EXP_GT_S1_GEN_S2_ROOT/v1.1/data/flood_events/HandLabeled/S2Hand/$new_filename"
done
```

The generated modalities are all the modalities that we don't use as condition.

E.g. in the above case they are
```
Condition Modalities:
  - S1RTC_vh_vv
Generate Modalities:
  - 3d_cartesian_lat_lon
  - DEM_DEM
  - LULC_LULC
  - S2L1C_B01_B09_B10
  - S2L1C_B02_B03_B04_B08
  - S2L1C_B05_B06_B07_B11_B12_B8A
  - S2L1C_cloud_mask
  - S2L2A_B01_B09
  - S2L2A_B02_B03_B04_B08
  - S2L2A_B05_B06_B07_B11_B12_B8A
  - mean_timestamps
```

Therefore, the output folder structure will be as follows:

```
./data/sen1floods11_v1.1/v1.1/outputs/input_S1_output_S2/
├── v1.1/
│   ├── data/
│   │   └── flood_events/
│   │       └── HandLabeled/
│   │           ├── 3d_cartesian_lat_lon/
│   │           │   ├── Ghana_103272_3d_cartesian_lat_lon.tif
│   │           │   ├── ...
│   │           ├── DEM_DEM/
│   │           │   ├── Ghana_103272_DEM_DEM.tif
│   │           │   ├── ...
│   │           ├── LULC_LULC/
│   │           │   ├── Ghana_103272_LULC_LULC.tif
│   │           │   ├── ...
│   │           ├── mean_timestamps/
│   │           │   ├── Ghana_103272_mean_timestamps.tif
│   │           │   ├── ...
│   │           ├── S2L1C/
│   │           │   ├── Ghana_103272_S2L1C.tif
│   │           │   ├── ...
│   │           ├── S2L1C_cloud_mask/
│   │           │   ├── Ghana_103272_S2L1C_cloud_mask.tif
│   │           │   ├── ...
│   │           └── S2L2A/
│   │               ├── Ghana_103272_S2L2A.tif
│   │               ├── ...
│   ├── Sen1Floods11_Metadata.geojson
│   └── splits/
│       ├── flood_handlabeled/
│       │   ├── flood_bolivia_data.csv
│       │   ├── flood_test_data.csv
│       │   ├── flood_train_data.csv
│       │   └── flood_valid_data.csv
│       └── perm_water/
│           ├── permanent_water_data.csv
│           ├── permanent_water_test_data.csv
│           ├── permanent_water_train_data.csv
│           └── permanent_water_validation_data.csv
└── visualisations/
    ├── 3d_cartesian_lat_lon/
    ├── DEM_DEM/
    ├── LULC_LULC/
    ├── mean_timestamps/
    ├── merged_visualisations/
    ├── S1RTC_vh_vv/
    ├── S2L1C_B01_B09_B10/
    ├── S2L1C_B02_B03_B04_B08/
    ├── S2L1C_B05_B06_B07_B11_B12_B8A/
    ├── S2L1C_cloud_mask/
    ├── S2L2A_B01_B09/
    ├── S2L2A_B02_B03_B04_B08/
    └── S2L2A_B05_B06_B07_B11_B12_B8A/
```



### Generate S1 from S2

Generate S1 from S2:

```bash
export EXP_GT_S2_GEN_S1_ROOT=$SEN1FLOODS11_ROOT/outputs/GT_S2_GEN_S1

python3 legacy_evaluations/sen1floods11/generate_sen1floods11.py \
    --model_path models/copgen/world_12_modalities_large/300000.ckpt/nnet_ema.pth \
    --model_config configs/copgen/discrete/cop_gen_large.py \
    --data_config configs/legacy_evaluations/sen1floods11/copgen_sen1floods11.yaml \
    --condition_modalities S2L2A_B01_B09 S2L2A_B02_B03_B04_B08 S2L2A_B05_B06_B07_B11_B12_B8A S2L1C_B01_B09_B10 S2L1C_B02_B03_B04_B08 S2L1C_B05_B06_B07_B11_B12_B8A \
    --root_output_path $EXP_GT_S2_GEN_S1_ROOT \
    --split train \
    --batch_size 16 \
    --visualise_every_n_batches 10
```

Make the necessary symbolic links to the ground truth labels and S2, and change the file naming convention of the generated S1.

```bash
# Link 192x192 GT Labels and GT S2 to experiment output folder
ln -s $SEN1FLOODS11_DATA_192_DATA_ROOT/LabelHand $EXP_GT_S2_GEN_S1_ROOT/v1.1/data/flood_events/HandLabeled/LabelHand

# Link 192x192 GT S2 to experiment output folder
ln -s $SEN1FLOODS11_DATA_192_DATA_ROOT/S2Hand $EXP_GT_S2_GEN_S1_ROOT/v1.1/data/flood_events/HandLabeled/S2Hand

# Copy generated S1 to correct naming convention
mkdir -p $EXP_GT_S2_GEN_S1_ROOT/v1.1/data/flood_events/HandLabeled/S1Hand
for tif in $EXP_GT_S2_GEN_S1_ROOT/v1.1/data/flood_events/HandLabeled/S1RTC/*.tif; do
    filename=$(basename "$tif")
    new_filename="${filename/S1RTC/S1Hand}"
    cp "$tif" "$EXP_GT_S2_GEN_S1_ROOT/v1.1/data/flood_events/HandLabeled/S1Hand/$new_filename"
done
```