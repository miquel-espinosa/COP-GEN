import os
import sys
import pyarrow.parquet as pq
import pandas as pd
import json
from pathlib import Path
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Extract all content from a parquet file')
    parser.add_argument('--parquet-file', type=str, required=True,
                        help='Name of the parquet file to extract')
    parser.add_argument('--output-dir', type=str, default='./extracted_data',
                        help='Directory to save extracted data (default: ./extracted_data)')
    return parser.parse_args()

def extract_parquet_content(parquet_path, output_dir):
    """Extract all content from a parquet file and save it to the output directory"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Extracting data from {parquet_path} to {output_dir}")
    
    # Open the parquet file
    pf = pq.ParquetFile(parquet_path)
    print(f"File contains {pf.num_row_groups} row groups")
    
    # Process each row group
    for rg_idx in range(pf.num_row_groups):
        print(f"\nProcessing row group {rg_idx+1}/{pf.num_row_groups}")
        
        # Read the row group
        table = pf.read_row_group(rg_idx)
        df = table.to_pandas()
        
        # Create a directory for this row group
        if pf.num_row_groups > 1:
            rg_dir = output_dir / f"row_group_{rg_idx}"
        else:
            rg_dir = output_dir
        rg_dir.mkdir(exist_ok=True)
        
        # Get metadata to create more meaningful directory names if possible
        product_id = df['product_id'][0] if 'product_id' in df.columns else f"sample_{rg_idx}"
        grid_cell = df['grid_cell'][0] if 'grid_cell' in df.columns else ""
        
        # Create a more descriptive directory name if possible
        sample_dir = rg_dir / f"{grid_cell}_{product_id}" if grid_cell else rg_dir / product_id
        sample_dir.mkdir(exist_ok=True)
        
        # Extract and save metadata to JSON
        metadata = {}
        for col in df.columns:
            if df[col].dtype != 'object' or (len(df[col]) > 0 and not isinstance(df[col].iloc[0], bytes)):
                # Convert non-binary data to JSON-serializable format
                try:
                    if col == 'timestamp' and pd.api.types.is_datetime64_any_dtype(df[col]):
                        metadata[col] = df[col].iloc[0].strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        value = df[col].iloc[0]
                        # Handle numpy types
                        if hasattr(value, 'item'):
                            metadata[col] = value.item()
                        else:
                            metadata[col] = value
                except Exception as e:
                    metadata[col] = f"Error converting: {str(e)}"
        
        # Save metadata
        with open(sample_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Extract and save binary data
        binary_columns = []
        for col in df.columns:
            if df[col].dtype == 'object' and len(df[col]) > 0 and isinstance(df[col].iloc[0], bytes):
                binary_columns.append(col)
                binary_data = df[col].iloc[0]
                
                # Determine file extension based on common column naming conventions
                if col == 'thumbnail':
                    extension = '.png'
                elif col.startswith('B') and col[1:].isdigit():  # Sentinel-2 bands
                    extension = '.tif'
                elif col in ['vv', 'vh']:  # Sentinel-1 bands
                    extension = '.tif'
                elif col == 'DEM':  # DEM data
                    extension = '.tif'
                elif col == 'cloud_mask':
                    extension = '.tif'
                else:
                    extension = '.bin'  # Generic binary data
                
                # Save binary data
                file_path = sample_dir / f"{col}{extension}"
                with open(file_path, "wb") as f:
                    f.write(binary_data)
                print(f"  Saved {col}{extension}, size: {len(binary_data)/1024:.1f} KB")
        
        print(f"  Extracted metadata and {len(binary_columns)} binary files to {sample_dir}")

def main():
    args = parse_args()
    parquet_path = Path(args.parquet_file)
    
    if not parquet_path.exists():
        print(f"Error: File {parquet_path} not found")
        sys.exit(1)
    
    # Extract all content
    extract_parquet_content(parquet_path, args.output_dir)
    print("\nExtraction complete!")

if __name__ == "__main__":
    main()
