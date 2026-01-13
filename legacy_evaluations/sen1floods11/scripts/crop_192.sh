#!/bin/bash

# Define paths
export SEN1FLOODS11_ROOT=$(pwd)/data/sen1floods11_v1.1/v1.1
export SEN1FLOODS11_DATA_192_ROOT=$SEN1FLOODS11_ROOT/data_192/v1.1
export SEN1FLOODS11_DATA_192_DATA_ROOT=$SEN1FLOODS11_DATA_192_ROOT/data/flood_events/HandLabeled
# Copy splits and metadata
mkdir -p $SEN1FLOODS11_DATA_192_DATA_ROOT
cp -r $SEN1FLOODS11_ROOT/splits $SEN1FLOODS11_DATA_192_ROOT/splits
cp $SEN1FLOODS11_ROOT/Sen1Floods11_Metadata.geojson $SEN1FLOODS11_DATA_192_ROOT/Sen1Floods11_Metadata.geojson
# Center-crop
python3 copgen_eval/sen1floods11/center_crop_sen1floods11.py --size 192 \
  --src-root $SEN1FLOODS11_ROOT/data/flood_events/HandLabeled \
  --dst-root $SEN1FLOODS11_DATA_192_DATA_ROOT