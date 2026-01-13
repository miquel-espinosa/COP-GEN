#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x.%j.out
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH -x gpuhost015,gpuhost004,gpuhost016
#SBATCH --cpus-per-task=32
#SBATCH --job-name=fast_copy_DEM
#SBATCH --mem=400G
##------------------------ End job description ------------------------
module purge && source /gws/nopw/j04/sensecdt/users/mespi/virtualenvs/triffuser/bin/activate

src_dir="/gws/nopw/j04/sensecdt/data/internal/majorTOM/world/Core-DEM-tar-gz"
dst_dir="/work/scratch-pw2/mespi/majorTOM/world/Core-DEM-tar-gz"

mkdir -p "$dst_dir"

# Create a list of files that need copying or re-copying
tmp_list=$(mktemp)

find "$src_dir" -maxdepth 1 -name '*.tar.gz' | while read -r src_file; do
  filename=$(basename "$src_file")
  dst_file="$dst_dir/$filename"

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

  echo "$src_file" >> "$tmp_list"
done

# Copy needed files in parallel (32 concurrent processes)
cat "$tmp_list" | xargs -P 64 -I{} cp -v {} "$dst_dir/"

# Cleanup
rm "$tmp_list"

# Finally copy the .cache file
cp /gws/nopw/j04/sensecdt/data/internal/majorTOM/world/Core-DEM-tar-gz/.cache_DEM.pkl /work/scratch-pw2/mespi/majorTOM/world/Core-DEM-tar-gz/.cache_DEM.pkl