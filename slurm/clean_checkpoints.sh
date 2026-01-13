#!/bin/bash

CKPT_DIR="${1:-.}"

# Get sorted list of checkpoint folder names (numerically)
checkpoints=($(find "$CKPT_DIR" -maxdepth 1 -type d -name "*.ckpt" \
  | sed -E 's#.*/([0-9]+)\.ckpt$#\1#' | sort -n))

echo "Found ${#checkpoints[@]} checkpoints"

# Find last two checkpoints
last1=${checkpoints[-1]}
last2=${checkpoints[-2]}

for ckpt in "${checkpoints[@]}"; do
  # Keep if multiple of 10000 or one of the last two
  if (( ckpt % 10000 == 0 )) || [[ "$ckpt" == "$last1" ]] || [[ "$ckpt" == "$last2" ]]; then
    continue
  fi
  # Delete folder
  echo "Deleting $ckpt.ckpt"
  rm -rf "$CKPT_DIR/$ckpt.ckpt"
done
