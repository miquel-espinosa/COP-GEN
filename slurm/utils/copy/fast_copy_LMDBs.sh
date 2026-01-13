#!/bin/bash

# Check if both arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <source_dir> <destination_dir>"
    exit 1
fi

src_dir="$1"
dst_dir="$2"

mkdir -p "$dst_dir"

# Navigate to the source directory so we can work with *relative* paths. This lets us
# recreate the original folder hierarchy (up to maxdepth 2) inside $dst_dir.
pushd "$src_dir" > /dev/null

# Create a list of files that need copying or re-copying (NULL-separated for safety)
tmp_list=$(mktemp)

find -L . -maxdepth 2 -name '*.mdb' -print0 | while IFS= read -r -d '' rel_path; do
  src_file="$src_dir/$rel_path"
  dst_file="$dst_dir/$rel_path"
  filename="$rel_path"

  # Ensure the destination sub-directory exists
  mkdir -p "$(dirname "$dst_file")"

  if [ -f "$dst_file" ]; then
    src_size=$(stat -c %s "$src_file")
    dst_size=$(stat -c %s "$dst_file")

    if [ "$src_size" -eq "$dst_size" ]; then
      echo "✅ Skipping: $filename"
      continue
    else
      echo "⚠️  Re-copying (partial): $filename"
      rm -f "$dst_file"
    fi
  else
    echo "📥 Copying new file: $filename"
  fi

  # Queue the file (relative path) for parallel copy
  printf '%s\0' "$rel_path" >> "$tmp_list"
done

popd > /dev/null

# Copy needed files in parallel (32 concurrent processes) while preserving the folder structure
pushd "$src_dir" > /dev/null
cat "$tmp_list" | xargs -0 -P 32 -I{} cp -v -L --parents "{}" "$dst_dir/"
popd > /dev/null

# Cleanup
rm "$tmp_list"
