import pandas as pd
import pyarrow.parquet as pq
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Read metadata.parquet and print download URLs')
    parser.add_argument('--metadata-path', type=str, required=True,
                        help='Path to the metadata.parquet file')
    return parser.parse_args()

def main():
    args = parse_args()
    metadata_path = Path(args.metadata_path)
    
    # Read the parquet file
    print(f"Reading metadata from: {metadata_path}")
    df = pq.read_table(metadata_path).to_pandas()
    
    # Extract unique parquet URLs
    unique_urls = df['parquet_url'].unique()
    
    # Print the URLs
    print(f"\nFound {len(unique_urls)} unique parquet file URLs:")
    for url in unique_urls:
        print(url)
    
    print(f"\nTotal number of samples in metadata: {len(df)}")

if __name__ == "__main__":
    main()
