import os
import sys
import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Extract information from a parquet file')
    parser.add_argument('--parquet-file', type=str, required=True,
                        help='Name of the parquet file in the current directory')
    parser.add_argument('--row-group', type=int, default=None,
                        help='Specific row group to extract (default: all row groups)')
    parser.add_argument('--sample-binary', action='store_true',
                        help='Print sample of binary content (first 100 bytes)')
    return parser.parse_args()

def main():
    args = parse_args()
    parquet_path = Path(args.parquet_file)
    
    if not parquet_path.exists():
        print(f"Error: File {parquet_path} not found")
        sys.exit(1)
    
    print(f"\n--- Analyzing parquet file: {parquet_path} ---\n")
    
    # Open the parquet file
    pf = pq.ParquetFile(parquet_path)
    
    # Print basic file information
    print(f"File size: {parquet_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"Number of row groups: {pf.num_row_groups}")
    print(f"Number of rows: {pf.metadata.num_rows}")
    print(f"Number of columns: {len(pf.schema_arrow)}")
    
    # Print schema information
    print("\nSchema:")
    for i, field in enumerate(pf.schema_arrow):
        print(f"  {i+1}. {field.name}: {field.type}")
    
    # Process row groups
    row_groups = [args.row_group] if args.row_group is not None else range(pf.num_row_groups)
    
    for rg_idx in row_groups:
        if rg_idx >= pf.num_row_groups:
            print(f"Error: Row group {rg_idx} does not exist (max: {pf.num_row_groups-1})")
            continue
            
        print(f"\n--- Row Group {rg_idx} ---")
        # Get row group metadata
        rg_metadata = pf.metadata.row_group(rg_idx)
        print(f"Row count: {rg_metadata.num_rows}")
        
        # Read the row group
        table = pf.read_row_group(rg_idx)
        df = table.to_pandas()
        
        # Display information about each column
        print("\nColumn information:")
        for col_name in df.columns:
            col_data = df[col_name]
            dtype = col_data.dtype
            
            if dtype == 'object':
                # Check if it's binary data
                if len(col_data) > 0 and isinstance(col_data.iloc[0], bytes):
                    item_size = len(col_data.iloc[0])
                    print(f"  {col_name}: Binary data, size: {item_size / 1024:.2f} KB")
                    
                    if args.sample_binary and item_size > 0:
                        print(f"    Sample (first 100 bytes): {col_data.iloc[0][:100]}")
                else:
                    # For non-binary object columns
                    print(f"  {col_name}: Object type, example: {col_data.iloc[0]}")
            else:
                # For numeric or other columns
                if col_data.size > 0:
                    print(f"  {col_name}: {dtype}, min: {col_data.min()}, max: {col_data.max()}, example: {col_data.iloc[0]}")
                else:
                    print(f"  {col_name}: {dtype}, empty column")
        
        # Print specific metadata fields for Major-TOM dataset
        if 'product_id' in df.columns:
            print(f"\nProduct ID: {df['product_id'].iloc[0]}")
        if 'grid_cell' in df.columns:
            print(f"Grid Cell: {df['grid_cell'].iloc[0]}")
        if 'timestamp' in df.columns:
            print(f"Timestamp: {df['timestamp'].iloc[0]}")

if __name__ == "__main__":
    main()
