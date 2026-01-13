import argparse
from pathlib import Path
from metadata_helpers import metadata_from_url, filter_download
import sys

def main():
    parser = argparse.ArgumentParser(description='Download specific Sentinel-2 scene or entire grid cell from Major-TOM on HuggingFace')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., Core-S2L1C)')
    
    # Create mutually exclusive group for scene_path and grid_cell
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--scene_path', type=str,
                        help='Path to specific scene like 536U/536U_904R/S2A_MSIL1C_20200216T024801_N0500_R132_T51UVP_20230501T220848')
    group.add_argument('--grid_cell', type=str,
                        help='Grid cell to download entirely (e.g., 536U)')
    
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save downloaded files')
    parser.add_argument('--tif_columns', type=str, nargs='*', 
                        help='Optional list of TIF columns to download (e.g., B04 B03 B02). If not specified, downloads all bands.')
    args = parser.parse_args()

    dataset_name = args.dataset
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup metadata URL
    hf_dataset_path = f'Major-TOM/{dataset_name}'
    parquet_url = f'https://huggingface.co/datasets/{hf_dataset_path}/resolve/main/metadata.parquet?download=true'
    local_parquet_path = output_dir / f'{dataset_name}.parquet'

    print(f"🔍 Loading metadata for dataset {dataset_name} ...")
    gdf = metadata_from_url(parquet_url, local_parquet_path)

    if args.scene_path:
        # Parse scene path for specific scene download
        try:
            grid_cell = args.scene_path.split('/')[1]  # e.g., 536U_904R
            product_id = args.scene_path.split('/')[-1]
        except IndexError:
            print("❌ Invalid scene path format. Expected format: <region>/<grid_cell>/<product_id>")
            sys.exit(1)

        # Filter metadata for the exact product
        match = gdf[(gdf['grid_cell'] == grid_cell) & (gdf['product_id'] == product_id)]

        if match.empty:
            print(f"❌ No match found for {product_id} in {grid_cell}")
            sys.exit(1)

        print(f"✅ Found matching product. Downloading data to {output_dir} ...")
        
    elif args.grid_cell:
        # Filter metadata for all products in the grid cell
        # Grid cell format like "536U" needs to match grid_cell patterns that start with it
        match = gdf[gdf['grid_cell'].str.startswith(args.grid_cell)]

        if match.empty:
            print(f"❌ No products found for grid cell {args.grid_cell}")
            sys.exit(1)

        print(f"✅ Found {len(match)} products in grid cell {args.grid_cell}. Downloading data to {output_dir} ...")

    # Download the filtered row(s)
    filter_download(match, local_dir=str(output_dir), source_name=dataset_name.split('/')[-1], by_row=True, tif_columns=args.tif_columns)

    print("🎉 Download complete.")

if __name__ == '__main__':
    main()
