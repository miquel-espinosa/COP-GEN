#!/bin/bash

# Script to flatten nested directories after tar.gz extraction
# Usage: ./flatten_directories.sh <target_directory>
#
# After tar.gz extraction, the directories are nested.
# For example:
#    A / A / A_A_A.tif
#    A / A / A_A_B.tif
#    A / A / A_A_C.tif
# This script flattens them into:
#    A / A_A_A.tif
#    A / A_A_B.tif
#    A / A_A_C.tif

# Check if a directory path was provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <target_directory>"
    echo "Example: $0 /path/to/extracted/files"
    exit 1
fi

TARGET_DIR="$1"

# Check if the target directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory '$TARGET_DIR' does not exist."
    exit 1
fi

# Change to the target directory
cd "$TARGET_DIR" || {
    echo "Error: Cannot change to directory '$TARGET_DIR'"
    exit 1
}

echo "Flattening directories in: $TARGET_DIR"

# Go through each subdirectory in the target folder
for dir in */; do
    # Skip if no directories found
    [ -d "$dir" ] || continue
    
    # Strip trailing slash
    dir=${dir%/}

    # Check if the sub-subfolder exists (e.g., A/A)
    if [ -d "$dir/$dir" ]; then
        echo "Flattening: $dir/$dir -> $dir"
        
        # Make a temporary location
        mkdir -p tmp_move

        # Move the inner contents to tmp
        mv "$dir/$dir" tmp_move/

        # Remove the now-empty outer directory
        rmdir "$dir"

        # Move the contents back to top level
        mv tmp_move/"$dir" ./

        # Clean up tmp
        rmdir tmp_move
    else
        echo "Skipping: $dir (no nested folder named $dir)"
    fi
done

echo "Flattening complete!"
