#!/bin/bash

# Default value for force flag
FORCE_COPY=false

# Parse options
while [[ "$1" == -* ]]; do
    case "$1" in
        -f|--force)
            FORCE_COPY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if at least 2 arguments are provided (subset name and at least one source)
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 [-f|--force] SUBSET_NAME SOURCE1 [SOURCE2 ...]"
    echo "Options:"
    echo "  -f, --force    Force copy even if destination directory exists"
    echo "Example: $0 small_world Core-S2L2A Core-S1GRD"
    echo "Example with force: $0 -f small_world Core-S2L2A Core-S1GRD"
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
    # Check if the source directory already exists in TMP_DIR and if we should force copy
    if [ ! -d "$TMP_DIR/$source" ] || [ "$FORCE_COPY" = true ]; then
        # If directory exists and force is enabled, remove it first
        if [ -d "$TMP_DIR/$source" ] && [ "$FORCE_COPY" = true ]; then
            echo "Force flag enabled: Removing existing directory $TMP_DIR/$source"
            rm -rf "$TMP_DIR/$source"
        fi
        
        echo "Copying and extracting $source..."
        cp "$SCRATCH_PATH/$source.tar.gz" "$TMP_DIR/"
        tar -xzf "$TMP_DIR/$source.tar.gz" -C "$TMP_DIR/"
        rm "$TMP_DIR/$source.tar.gz"
    else
        echo "Directory $TMP_DIR/$source already exists, skipping..."
    fi
done