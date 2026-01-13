#!/bin/bash

# Check if dataset folder argument is provided
if [ $# -eq 0 ]; then
    echo "Error: Dataset folder path is required"
    echo "Usage: $0 <dataset_folder_path>"
    echo "Example: $0 paper_figures/one_tile_datasets_DEM_to_S2L2A"
    exit 1
fi

DATASET_FOLDER="$1"

# Check if the provided path exists and is a directory
if [ ! -d "$DATASET_FOLDER" ]; then
    echo "Error: '$DATASET_FOLDER' is not a valid directory"
    exit 1
fi

for exp in "$DATASET_FOLDER"/*; do
    # Skip if not a directory
    [ -d "$exp" ] || continue

    outputs_dir="$exp/outputs"

    # Skip if outputs folder missing
    [ -d "$outputs_dir" ] || continue

    base_name_experiment=$(basename "$exp")

    echo
    echo "=========================================="
    echo "Experiment: $base_name_experiment"
    echo "=========================================="

    cd "$outputs_dir"

    #################################
    # COPGEN
    #################################
    if [ -d "copgen" ]; then
        for copgen_run in copgen/*; do
            [ -d "$copgen_run" ] || continue

            # Extract seed from directory name
            seed=$(basename "$copgen_run" | grep -oP 'seed_\K[0-9]+')
            
            if [ -z "$seed" ]; then
                continue
            fi

            copgen_src="$copgen_run/visualisations/S2L2A_B02_B03_B04_B08/visualisations-0/composite_0_1_2_raw.png"
            # copgen_src="$copgen_run/visualisations/S2L2A_B02_B03_B04_B08/visualisations-0/band_0_raw.png"
            copgen_metrics="$copgen_run/output_metrics.txt"

            # Extract CopGen MAE
            if [ -f "$copgen_metrics" ]; then
                copgen_mae=$(grep "\[S2L2A\]" "$copgen_metrics" | sed -E 's/.*mae=([0-9.]+).*/\1/')
            else
                copgen_mae="UNKNOWN"
            fi

            echo "CopGen     seed=$seed  MAE=$copgen_mae"

            # Destination filename including MAE
            copgen_dst="copgen/S2L2A_seed_${seed}_mae_${copgen_mae}.png"

            # Copy PNG
            if [ -f "$copgen_src" ]; then
                cp "$copgen_src" "$copgen_dst"
            fi
        done
    fi

    #################################
    # TERRAMIND
    #################################
    first_terr_done=false

    if [ -d "terramind" ]; then
        for terr_run in terramind/*; do
            [ -d "$terr_run" ] || continue

            # Extract seed from directory name
            seed=$(basename "$terr_run" | grep -oP 'seed_\K[0-9]+')
            [ -z "$seed" ] && continue

            terr_vis_dir="$terr_run/visualisations/${base_name_experiment}"
            terr_src="$terr_vis_dir/pred_S2L2A.png"
            terr_metrics="$terr_run/output_metrics.txt"

            # Extract Terramind MAE
            if [ -f "$terr_metrics" ]; then
                terr_mae=$(grep "\[S2L2A\]" "$terr_metrics" | sed -E 's/.*mae=([0-9.]+).*/\1/')
            else
                terr_mae="UNKNOWN"
            fi

            echo "TerraMind  seed=$seed  MAE=$terr_mae"

            # Destination filename including MAE
            terr_dst="terramind/S2L2A_seed_${seed}_mae_${terr_mae}.png"

            # Copy pred_S2L2A
            if [ -f "$terr_src" ]; then
                cp "$terr_src" "$terr_dst"

                # ---- Create 192x192 crop for pred_S2L2A ----
                python3 - <<EOF
from PIL import Image
import os

src = "${terr_dst}"
out = src.replace(".png", "_crop192.png")

img = Image.open(src)
w, h = img.size
crop_w = crop_h = 192
left = (w - crop_w) // 2
top = (h - crop_h) // 2
img.crop((left, top, left+crop_w, top+crop_h)).save(out)
EOF
            fi

            ###########################################
            # FOR THE FIRST TERRAMIND FOLDER ONLY
            ###########################################
            if [ "$first_terr_done" = false ] && [ -d "$terr_vis_dir" ]; then
                echo "Processing ALL PNGs from first TerraMind folder: $terr_vis_dir"
                first_terr_done=true

                for png in "$terr_vis_dir"/*.png; do
                    [ -f "$png" ] || continue

                    base=$(basename "$png")
                    dst="terramind/${base}"

                    cp "$png" "$dst"

                    # ---- Crop the additional PNGs ----
                    python3 - <<EOF
from PIL import Image
import os

src = "${dst}"
out = src.replace(".png", "_crop192.png")

img = Image.open(src)
w, h = img.size
crop_w = crop_h = 192
left = (w - crop_w) // 2
top = (h - crop_h) // 2
img.crop((left, top, left+crop_w, top+crop_h)).save(out)
EOF
                done
            fi

        done
    fi



    cd - >/dev/null
done
