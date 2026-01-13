#!/bin/bash

# Usage: ./copy_and_extract.sh /path/to/source /path/to/destination

set -euo pipefail

SOURCE_DIR="$1"
DEST_DIR="$2"
TMP_DIR="$3"

# Check source directory
if [[ ! -d "$SOURCE_DIR" ]]; then
    echo "❌ Source directory does not exist: $SOURCE_DIR"
    exit 1
fi

# Ensure destination directory exists
mkdir -p "$DEST_DIR"

echo "🚀 Starting selective copy and extract from:"
echo "   Source: $SOURCE_DIR"
echo "   Dest:   $DEST_DIR"
echo

# Function to process a single file
process_file() {
    local src_file="$1"
    local dest_dir="$2"
    local file_name="$(basename "$src_file")"
    local base_name="${file_name%.tar.gz}"
    local dest_extract_path="$dest_dir/$base_name"

    # Skip if already extracted
    if [[ -d "$dest_extract_path" ]]; then
        echo "✅ Skipping $file_name (already extracted)"
        return
    fi

    echo "📦 Processing $file_name..."

    # Copy to temp first (preserving source), then extract
    local temp_file="$TMP_DIR/$file_name"

    echo "  🔄 Copying to $TMP_DIR..."
    rsync -a --progress "$src_file" "$temp_file"

    echo "  📂 Extracting to $dest_extract_path..."
    mkdir -p "$dest_extract_path"
    tar -xzf "$temp_file" -C "$dest_extract_path"

    echo "  🧹 Cleaning up $temp_file..."
    rm -f "$temp_file"

    echo "✅ Done with $file_name"
}

# Get number of CPU cores
NUM_CORES=$(nproc)
echo "Using $NUM_CORES concurrent processes"

# Loop through all .tar.gz files and process them in parallel
for src_file in "$SOURCE_DIR"/*.tar.gz; do
    # Wait if we've reached the maximum number of concurrent processes
    while [[ $(jobs -r | wc -l) -ge $NUM_CORES ]]; do
        sleep 1
    done

    # Process file in background
    process_file "$src_file" "$DEST_DIR" &
done

# Wait for all background processes to complete
wait

echo "🎉 All files processed."