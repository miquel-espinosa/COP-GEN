#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x.%j.out
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --gres=gpu:4
#SBATCH -w gpuhost012
#SBATCH --cpus-per-task=4
#SBATCH --job-name=copying
#SBATCH --mem=400G
##------------------------ End job description ------------------------

module purge && source /gws/nopw/j04/sensecdt/users/mespi/virtualenvs/triffuser/bin/activate

# Copy from work_scratch_out_images to tmp
rsync -az --info=progress2 --partial /home/users/mespi/projects/triffuser/work_scratch_out_images/test /tmp/majorTOM-tmp/world/word_final_test

# Copy from tmp to claptrap
rsync -az --info=progress2 --partial /tmp/majorTOM-tmp/world/word_final_test claptrap:/localdisk/home/s2254242/datasets/majorTOM/world/world_final_test


# echo "Not implemented yet"

# exit 0

# TMP_DIR="/tmp/majorTOM-tmp/world"
# SUFFIX_PNG="world_thumbnail_centercrop_png"
# SAVE_DIR_PNG="$TMP_DIR/$SUFFIX_PNG"
# SAVE_DIR="/gws/nopw/j04/sensecdt/data/internal/majorTOM/world/$SUFFIX_PNG"

# mkdir -p $SAVE_DIR/test/DEM_thumbnail
# mkdir -p $SAVE_DIR/test/S1RTC_thumbnail
# mkdir -p $SAVE_DIR/test/S2L1C_thumbnail
# mkdir -p $SAVE_DIR/test/S2L2A_thumbnail

# rsync -az --info=progress2 --partial "$SAVE_DIR_PNG/test/DEM_thumbnail/"    "$SAVE_DIR/test/DEM_thumbnail" &
# rsync -az --info=progress2 --partial "$SAVE_DIR_PNG/test/S1RTC_thumbnail/"  "$SAVE_DIR/test/S1RTC_thumbnail" &
# rsync -az --info=progress2 --partial "$SAVE_DIR_PNG/test/S2L1C_thumbnail/"  "$SAVE_DIR/test/S2L1C_thumbnail" &
# rsync -az --info=progress2 --partial "$SAVE_DIR_PNG/test/S2L2A_thumbnail/"  "$SAVE_DIR/test/S2L2A_thumbnail" &

# wait

# echo "Finished copying"



# CHECK SAMPLE

# mkdir gen
# mkdir real
# SAMPLE=390D_1325R_center
# cd gen
# cp ../../work_scratch_out_images/test_set_1/condition_s2l2a_generate_dem_s1rtc_s2l1c_1samples/dem/$SAMPLE.png ./demgen.png
# cp ../../work_scratch_out_images/test_set_1/condition_s2l2a_generate_dem_s1rtc_s2l1c_1samples/s2_l1c/$SAMPLE.png ./s2_l1cgen.png
# cp ../../work_scratch_out_images/test_set_1/condition_s2l2a_generate_dem_s1rtc_s2l1c_1samples/s1_rtc/$SAMPLE.png ./s1_rtcgen.png
# cd ../real
# cp ../../data/majorTOM/world/world_thumbnail_centercrop_png/test/DEM_thumbnail/$SAMPLE.png ./demreal.png
# cp ../../data/majorTOM/world/world_thumbnail_centercrop_png/test/S1RTC_thumbnail/$SAMPLE.png ./s1_rtc.png
# cp ../../data/majorTOM/world/world_thumbnail_centercrop_png/test/S2L1C_thumbnail/$SAMPLE.png ./s2_l1creal.png
# cp ../../data/majorTOM/world/world_thumbnail_centercrop_png/test/S2L2A_thumbnail/$SAMPLE.png ./s2_l2areal.png
# cd ..