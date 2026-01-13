"""
This script will delete the specified incorrect entries from a given dataset and the pickle file.
"""

import pickle
import os
import shutil
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Delete incorrect entries from a dataset directory and its cache pickle file."
)
parser.add_argument(
    "-s",
    "--satellite-name",
    required=True,
    help="Name of the satellite (e.g., S2L1C)",
)
parser.add_argument(
    "-i",
    "--incorrect-entries",
    nargs="+",
    required=True,
    help="Space-separated list of incorrect entry IDs (e.g., 536U_904R 208U_39L)",
)
parser.add_argument(
    "-d",
    "--dataset-path",
    required=True,
    help="Path to the dataset root directory (e.g., /tmp/majorTOM/world/Core-S2L1C)",
)

args = parser.parse_args()

SATELLITE_NAME = args.satellite_name
INCORRECT_ENTRIES = args.incorrect_entries
DATASET_PATH = args.dataset_path


# SATELLITE_NAME = 'S1RTC'
# INCORRECT_ENTRIES = ['208U_39L', '211U_29L', '210U_40L']
# DATASET_PATH = f'/work/scratch-pw3/mespi/majorTOM/world/Core-S1RTC' # Change path for other satellites

# Remove incorrect entries from the dataset
for entry in INCORRECT_ENTRIES:
    # Remove dir and all its contents
    grid_cell = entry.split('_')[0]
    # If the path exists, remove it
    if os.path.exists(os.path.join(DATASET_PATH, f"{grid_cell}", f"{entry}")):
        print(f'Removing {os.path.join(DATASET_PATH, f"{grid_cell}", f"{entry}")}')
        shutil.rmtree(os.path.join(DATASET_PATH, f'{grid_cell}', f'{entry}'))
    else:
        print(f'Path {os.path.join(DATASET_PATH, f"{grid_cell}", f"{entry}")} does not exist')

# Path to the pickle file
cache_file = f'.cache_{SATELLITE_NAME}.pkl'

# Load the pickle data
with open(f'{DATASET_PATH}/{cache_file}', 'rb') as f:
    data = pickle.load(f)

# Print length before deletion
print("Length before deletion:", len(data))

# Remove entry if it exists
data = [entry for entry in data if entry[0] not in INCORRECT_ENTRIES]

# Print length after deletion
print("Length after deletion:", len(data))

# Overwrite the pickle file
with open(f'{DATASET_PATH}/{cache_file}', 'wb') as f:
    pickle.dump(data, f)
