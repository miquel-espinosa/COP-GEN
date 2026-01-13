#!/bin/bash

# Check if at least 2 arguments are provided (subset name and at least one source)
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 SUBSET_NAME SOURCE1 [SOURCE2 ...]"
    echo "Example: $0 small_world Core-S2L2A Core-S1GRD"
    exit 1
fi

# Get subset name from first argument
SUBSET_NAME="$1"
shift  # Remove first argument from the arguments list

# Get sources from remaining arguments
SOURCES=("$@")

SCRATCH_PATH="/work/scratch-pw2/mespi/majorTOM/$SUBSET_NAME"
TMP_DIR="/tmp/majorTOM/$SUBSET_NAME"

# Create destination directory if it doesn't exist
mkdir -p "$TMP_DIR"

# Process each source
for source in "${SOURCES[@]}"; do
    # Check if the source directory already exists in TMP_DIR
    if [ ! -d "$TMP_DIR/$source" ]; then
        echo "Copying and extracting $source..."
        cp "$SCRATCH_PATH/$source.tar.gz" "$TMP_DIR/"
        tar -xzf "$TMP_DIR/$source.tar.gz" -C "$TMP_DIR/"
        rm "$TMP_DIR/$source.tar.gz"
    else
        echo "Directory $TMP_DIR/$source already exists, skipping..."
    fi
done