#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x.%j.out
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --gres=gpu:4
#SBATCH -w gpuhost011
#SBATCH --cpus-per-task=4
#SBATCH --job-name=inception
#SBATCH --mem=400G
##------------------------ End job description ------------------------

module purge && source /gws/nopw/j04/sensecdt/users/mespi/virtualenvs/triffuser/bin/activate

# rsync -az --info=progress2 --partial --ignore-existing /gws/nopw/j04/sensecdt/data/internal/majorTOM/world/world_thumbnail_centercrop_png/test/ /tmp/majorTOM-tmp/world/world_thumbnail_centercrop_png/test/

# #### Test set 1. Condition on S1 RTC, evaluate on DEM, S2 L1C, S2 L2A

# echo "Running ISC for test set 1. Condition on S1 RTC, evaluate on DEM"
# fidelity --gpu 0 --isc --input1 work_scratch_out_images/test_set_1/condition_s1rtc_generate_dem_s2l1c_s2l2a_1samples/dem

# echo "Running ISC for test set 1. Condition on S1 RTC, evaluate on S2 L1C"
# fidelity --gpu 0 --isc --input1 work_scratch_out_images/test_set_1/condition_s1rtc_generate_dem_s2l1c_s2l2a_1samples/s2_l1c

# echo "Running ISC for test set 1. Condition on S1 RTC, evaluate on S2 L2A"
# fidelity --gpu 0 --isc --input1 work_scratch_out_images/test_set_1/condition_s1rtc_generate_dem_s2l1c_s2l2a_1samples/s2_l2a


# #### Test set 1. Condition on S2 L2A, evaluate on DEM, S1 RTC, S2 L1C

# echo "Running ISC for test set 1. Condition on S2 L2A, evaluate on DEM"
# fidelity --gpu 0 --isc --input1 work_scratch_out_images/test_set_1/condition_s2l2a_generate_dem_s1rtc_s2l1c_1samples/dem

# echo "Running ISC for test set 1. Condition on S2 L2A, evaluate on S1 RTC"
# fidelity --gpu 0 --isc --input1 work_scratch_out_images/test_set_1/condition_s2l2a_generate_dem_s1rtc_s2l1c_1samples/s1_rtc

# echo "Running ISC for test set 1. Condition on S2 L2A, evaluate on S2 L1C"
# fidelity --gpu 0 --isc --input1 work_scratch_out_images/test_set_1/condition_s2l2a_generate_dem_s1rtc_s2l1c_1samples/s2_l1c


#### Real images

# echo "Running ISC for real images. Evaluating DEM"
# fidelity --gpu 0 --isc --input1 /gws/nopw/j04/sensecdt/data/internal/majorTOM/world/world_thumbnail_centercrop_png/test/DEM_thumbnail

# echo "Running ISC for real images. Evaluating S1 RTC"
# fidelity --gpu 0 --isc --input1 /gws/nopw/j04/sensecdt/data/internal/majorTOM/world/world_thumbnail_centercrop_png/test/S1RTC_thumbnail

# echo "Running ISC for real images. Evaluating S2 L1C"
# fidelity --gpu 0 --isc --input1 /gws/nopw/j04/sensecdt/data/internal/majorTOM/world/world_thumbnail_centercrop_png/test/S2L1C_thumbnail

# echo "Running ISC for real images. Evaluating S2 L2A"
# fidelity --gpu 0 --isc --input1 /gws/nopw/j04/sensecdt/data/internal/majorTOM/world/world_thumbnail_centercrop_png/test/S2L2A_thumbnail



# #### Test set 1. Unconditional generation

# echo "Running ISC for test set 1. Unconditional generation on DEM"
# fidelity --gpu 0 --isc --input1 work_scratch_out_images/test_set_1/generate_dem_s1rtc_s2l1c_s2l2a_1samples/dem

# echo "Running ISC for test set 1. Unconditional generation on S1 RTC"
# fidelity --gpu 0 --isc --input1 work_scratch_out_images/test_set_1/generate_dem_s1rtc_s2l1c_s2l2a_1samples/s1_rtc

# echo "Running ISC for test set 1. Unconditional generation on S2 L1C"
# fidelity --gpu 0 --isc --input1 work_scratch_out_images/test_set_1/generate_dem_s1rtc_s2l1c_s2l2a_1samples/s2_l1c

# echo "Running ISC for test set 1. Unconditional generation on S2 L2A"
# fidelity --gpu 0 --isc --input1 work_scratch_out_images/test_set_1/generate_dem_s1rtc_s2l1c_s2l2a_1samples/s2_l2a

# rsync -az --info=progress2 --partial /gws/nopw/j04/sensecdt/data/internal/majorTOM/world/world_thumbnail_centercrop_png/test /home/users/mespi/projects/triffuser/work_scratch_out_images


# #### Test set 1. Condition on DEM, evaluate on S1 RTC, S2 L1C, S2 L2A

# echo "Running ISC for test set 3. Condition on DEM, evaluate on S1 RTC"
# fidelity --gpu 0 --isc --input1 work_scratch_out_images/test_set_3/condition_dem_generate_s1rtc_s2l1c_s2l2a_1samples/s1_rtc

# echo "Running ISC for test set 3. Condition on DEM, evaluate on S2 L1C"
# fidelity --gpu 0 --isc --input1 work_scratch_out_images/test_set_3/condition_dem_generate_s1rtc_s2l1c_s2l2a_1samples/s2_l1c

# echo "Running ISC for test set 3. Condition on DEM, evaluate on S2 L2A"
# fidelity --gpu 0 --isc --input1 work_scratch_out_images/test_set_3/condition_dem_generate_s1rtc_s2l1c_s2l2a_1samples/s2_l2a


# # #### Test set 1. Condition on S2L1C, evaluate on DEM, S1 RTC, S2 L2A

# echo "Running ISC for test set 4. Condition on S2L1C, evaluate on DEM"
# fidelity --gpu 0 --isc --input1 work_scratch_out_images/test_set_4/condition_s2l1c_generate_dem_s1rtc_s2l2a_1samples/dem

# echo "Running ISC for test set 4. Condition on S2L1C, evaluate on S1 RTC"
# fidelity --gpu 0 --isc --input1 work_scratch_out_images/test_set_4/condition_s2l1c_generate_dem_s1rtc_s2l2a_1samples/s1_rtc

# echo "Running ISC for test set 4. Condition on S2L1C, evaluate on S2 L2A"
# fidelity --gpu 0 --isc --input1 work_scratch_out_images/test_set_4/condition_s2l1c_generate_dem_s1rtc_s2l2a_1samples/s2_l2a



### FID:


# Test set 3. Condition DEM. Generated S1RTC
# echo -e "\e[31mTest set 3. Condition DEM. Generated S1RTC\e[0m"
# fidelity --gpu 0 --fid --input1 work_scratch_out_images/test_set_3/condition_dem_generate_s1rtc_s2l1c_s2l2a_1samples/s1_rtc --input2 /home/users/mespi/test/S1RTC_thumbnail

# # Test set 3. Condition DEM. Generated S2L1C
# echo -e "\e[31mTest set 3. Condition DEM. Generated S2L1C\e[0m"
# fidelity --gpu 0 --fid --input1 work_scratch_out_images/test_set_3/condition_dem_generate_s1rtc_s2l1c_s2l2a_1samples/s2_l1c --input2 /home/users/mespi/test/S2L1C_thumbnail

# # Test set 3. Condition DEM. Generated S2L2A
# echo -e "\e[31mTest set 3. Condition DEM. Generated S2L2A\e[0m"
# fidelity --gpu 0 --fid --input1 work_scratch_out_images/test_set_3/condition_dem_generate_s1rtc_s2l1c_s2l2a_1samples/s2_l2a --input2 /home/users/mespi/test/S2L2A_thumbnail


# # Test set 1. Condition S1RTC. Generated DEM
# echo -e "\e[31mTest set 1. Condition S1RTC. Generated DEM\e[0m"
# fidelity --gpu 0 --fid --input1 work_scratch_out_images/test_set_1/condition_s1rtc_generate_dem_s2l1c_s2l2a_1samples/dem --input2 /home/users/mespi/test/DEM_thumbnail

# # Test set 1. Condition S1RTC. Generated S2L1C
# echo -e "\e[31mTest set 1. Condition S1RTC. Generated S2L1C\e[0m"
# fidelity --gpu 0 --fid --input1 work_scratch_out_images/test_set_1/condition_s1rtc_generate_dem_s2l1c_s2l2a_1samples/s2_l1c --input2 /home/users/mespi/test/S2L1C_thumbnail

# # Test set 1. Condition S1RTC. Generated S2L2A
# echo -e "\e[31mTest set 1. Condition S1RTC. Generated S2L2A\e[0m"
# fidelity --gpu 0 --fid --input1 work_scratch_out_images/test_set_1/condition_s1rtc_generate_dem_s2l1c_s2l2a_1samples/s2_l2a --input2 /home/users/mespi/test/S2L2A_thumbnail


# # Test set 4. Condition S2L1C. Generated DEM
# echo -e "\e[31mTest set 4. Condition S2L1C. Generated DEM\e[0m"
# fidelity --gpu 0 --fid --input1 work_scratch_out_images/test_set_4/condition_s2l1c_generate_dem_s1rtc_s2l2a_1samples/dem --input2 /home/users/mespi/test/DEM_thumbnail

# # Test set 4. Condition S2L1C. Generated S1RTC
# echo -e "\e[31mTest set 4. Condition S2L1C. Generated S1RTC\e[0m"
# fidelity --gpu 0 --fid --input1 work_scratch_out_images/test_set_4/condition_s2l1c_generate_dem_s1rtc_s2l2a_1samples/s1_rtc --input2 /home/users/mespi/test/S1RTC_thumbnail

# # Test set 4. Condition S2L1C. Generated S2L2A
# echo -e "\e[31mTest set 4. Condition S2L1C. Generated S2L2A\e[0m"
# fidelity --gpu 0 --fid --input1 work_scratch_out_images/test_set_4/condition_s2l1c_generate_dem_s1rtc_s2l2a_1samples/s2_l2a --input2 /home/users/mespi/test/S2L2A_thumbnail


# # Test set 1. Condition S2L2A. Generated DEM
# echo -e "\e[31mTest set 1. Condition S2L2A. Generated DEM\e[0m"
# fidelity --gpu 0 --fid --input1 work_scratch_out_images/test_set_1/condition_s2l2a_generate_dem_s1rtc_s2l1c_1samples/dem --input2 /home/users/mespi/test/DEM_thumbnail

# # Test set 1. Condition S2L2A. Generated S1RTC
# echo -e "\e[31mTest set 1. Condition S2L2A. Generated S1RTC\e[0m"
# fidelity --gpu 0 --fid --input1 work_scratch_out_images/test_set_1/condition_s2l2a_generate_dem_s1rtc_s2l1c_1samples/s1_rtc --input2 /home/users/mespi/test/S1RTC_thumbnail

# # Test set 1. Condition S2L2A. Generated S2L1C
# echo -e "\e[31mTest set 1. Condition S2L2A. Generated S2L1C\e[0m"
# fidelity --gpu 0 --fid --input1 work_scratch_out_images/test_set_1/condition_s2l2a_generate_dem_s1rtc_s2l1c_1samples/s2_l1c --input2 /home/users/mespi/test/S2L1C_thumbnail



# Unconditional generation
echo -e "\e[31mUnconditional generation. DEM\e[0m"
fidelity --gpu 0 --fid --input1 work_scratch_out_images/test_set_1/generate_dem_s1rtc_s2l1c_s2l2a_1samples/dem --input2 /home/users/mespi/test/DEM_thumbnail

echo -e "\e[31mUnconditional generation. S1RTC\e[0m"
fidelity --gpu 0 --fid --input1 work_scratch_out_images/test_set_1/generate_dem_s1rtc_s2l1c_s2l2a_1samples/s1_rtc --input2 /home/users/mespi/test/S1RTC_thumbnail

echo -e "\e[31mUnconditional generation. S2L1C\e[0m"
fidelity --gpu 0 --fid --input1 work_scratch_out_images/test_set_1/generate_dem_s1rtc_s2l1c_s2l2a_1samples/s2_l1c --input2 /home/users/mespi/test/S2L1C_thumbnail

echo -e "\e[31mUnconditional generation. S2L2A\e[0m"
fidelity --gpu 0 --fid --input1 work_scratch_out_images/test_set_1/generate_dem_s1rtc_s2l1c_s2l2a_1samples/s2_l2a --input2 /home/users/mespi/test/S2L2A_thumbnail

