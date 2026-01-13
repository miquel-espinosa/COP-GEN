"""
Usage:

python3 paper_figures/crop_stacked_narrow_distribution.py ./paper_figures/paper_figures_datasets/one_tile_distribution_narrowing/195D_669L/narrowing_histograms
"""


import os
import argparse
from PIL import Image

# Parse command line arguments
parser = argparse.ArgumentParser(description="Crop stacked PNG files")
parser.add_argument("input_dir", type=str, help="Directory containing PNG files to crop")
args = parser.parse_args()

# Directory containing your PNG files
input_dir = args.input_dir
output_dir = os.path.join(input_dir, "cropped")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Crop parameters
# x = 28
x = 121
y = 264
# width = 2325
width = 2233
height = 4490

crop_box = (x, y, x + width, y + height)

# Loop through files
for filename in os.listdir(input_dir):
    if filename.endswith("_stacked.png"):
        input_path = os.path.join(input_dir, filename)

        # Open image
        img = Image.open(input_path)

        # Crop
        cropped = img.crop(crop_box)

        # Save to new folder
        output_path = os.path.join(output_dir, filename)
        cropped.save(output_path)

        print(f"Cropped: {filename}")

print("Done!")
