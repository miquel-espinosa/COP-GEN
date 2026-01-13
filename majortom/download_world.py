import argparse
from shapely.geometry import box
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import geopandas as gpd
import pandas as pd
import pyarrow.parquet as pq
from typing import List, Dict, Set
import logging
import urllib.request
from concurrent import futures
import fsspec
from tqdm import tqdm
import tempfile
import time
import random


S2L2A_METADATA = ['grid_cell', 'grid_row_u', 'grid_col_r', 'product_id', 'timestamp', 'cloud_cover', 'nodata', 'centre_lat', 'centre_lon', 'crs', 'parquet_url', 'parquet_row', 'geometry']
S2L1C_METADATA = ['grid_cell', 'grid_row_u', 'grid_col_r', 'product_id', 'timestamp', 'cloud_cover', 'nodata', 'centre_lat', 'centre_lon', 'crs', 'parquet_url', 'parquet_row', 'geometry']
S1RTC_METADATA = ['grid_cell', 'grid_row_u', 'grid_col_r', 'product_id', 'timestamp', 'nodata', 'orbit_state', 'centre_lat', 'centre_lon', 'crs', 'parquet_url', 'parquet_row']
DEM_METADATA = ['grid_cell', 'grid_row_u', 'grid_col_r', 'nodata', 'max_val', 'min_val', 'centre_lat', 'centre_lon', 'crs', 'parquet_url', 'parquet_row', '__index_level_0__']

METADATA_COLUMNS = {
    'Core-S2L2A': S2L2A_METADATA,
    'Core-S2L1C': S2L1C_METADATA,
    'Core-S1RTC': S1RTC_METADATA,
    'Core-DEM': DEM_METADATA
}

CONTENT = {
    'Core-S2L2A': ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12', 'cloud_mask'],
    'Core-S2L1C': ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12', 'cloud_mask'],
    'Core-S1RTC': ['vv', 'vh'],
    'Core-DEM': ['DEM', 'compressed']
}

# Default max workers for extraction (can be higher as it's CPU-bound)
MAX_WORKERS = 32
# Default max workers for download (more conservative to avoid network issues)
DEFAULT_DOWNLOAD_WORKERS = 8

def parse_args():
    
    if "INTERACTIVE" in os.environ:  # Set INTERACTIVE=1 when running manually
        return argparse.Namespace(
            data_dir="./data/majorTOM",
            bbox=[-180.0, -90.0, 180.0, 90.0],
            sources=['Core-S2L2A', 'Core-S2L1C', 'Core-S1RTC', 'Core-DEM'],
            subset_name="world",
            start_date="2017-01-01",
            end_date="2025-01-01",
            cloud_cover=[0, 10],
            preview=True,
            mode="full",
            delete_parquets=False,
            download_workers=DEFAULT_DOWNLOAD_WORKERS,
            revalidate=False
        )
    else:
        parser = argparse.ArgumentParser(description='Download satellite imagery from Major-TOM dataset')
        parser.add_argument('--data-dir', type=str, default='./data/majorTOM',
                        help='Data directory for downloaded files')
        parser.add_argument('--extract-to-dir', type=str, default=None,
                        help='Directory to extract the downloaded files to')
        parser.add_argument('--bbox', type=float, nargs=4,
                        default=[2.9559111595, 43.8179931641, 55.4920501709, 65.808380127],
                        help='Bounding box coordinates: minx miny maxx maxy')
        parser.add_argument('--sources', type=str, nargs='+',
                        default=['Core-DEM', 'Core-S2L2A', 'Core-S2L1C', 'Core-S1RTC'],
                        help='List of source names for the datasets')
        parser.add_argument('--subset-name', type=str, required=True,
                        help='Name for the geographical subset being created')
        parser.add_argument('--start-date', type=str, default='2017-01-01',
                        help='Start date for temporal range (YYYY-MM-DD)')
        parser.add_argument('--end-date', type=str, default='2025-01-01',
                        help='End date for temporal range (YYYY-MM-DD)')
        parser.add_argument('--cloud-cover', type=float, nargs=2, default=[0, 10],
                        help='Cloud cover range (min max)')
        parser.add_argument('--criteria', type=str, default=None,
                        help='Criteria for timestamp deduplication. Currently we support "latest"')
        parser.add_argument('--n-samples', type=int, default=None,
                        help='Number of samples to download')
        parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
        parser.add_argument('--modalities', type=str, nargs='+', default=None,
                        help='List of modalities to process/download')
        parser.add_argument('--preview', action='store_true',
                        help='If True, only print the number of samples for each source that will be downloaded')
        parser.add_argument('--mode', type=str, choices=['full', 'download', 'extract'], default='full',
                        help='Mode of operation: full (download and extract), download (download parquets only), extract (extract from downloaded parquets)')
        parser.add_argument('--delete-parquets', action='store_true',
                        help='Delete parquet files after extraction (only used with extract mode)')
        parser.add_argument('--download-workers', type=int, default=DEFAULT_DOWNLOAD_WORKERS,
                        help=f'Number of parallel workers for downloading files. Default: {DEFAULT_DOWNLOAD_WORKERS}. Reduce this number if downloads are slow.')
        parser.add_argument('--revalidate', action='store_true',
                        help='Force revalidation of all parquet files and redownload if corrupted')
        return parser.parse_args()

    
def fix_crs(df):
    if df['crs'].iloc[0].startswith('EPSG:EPSG:'):
        df['crs'] = df['crs'].str.replace('EPSG:EPSG:', 'EPSG:', regex=False)
    return df

def my_filter_metadata(df,
                    region=None,
                    daterange=None,
                    cloud_cover=(0,100),
                    nodata=(0, 1.0)
                   ):
    """Filters the Major-TOM dataframe based on several parameters

    Args:
        df (geopandas dataframe): Parent dataframe
        region (shapely geometry object) : Region of interest
        daterange (tuple) : Inclusive range of dates (example format: '2020-01-01')
        cloud_cover (tuple) : Inclusive percentage range (0-100) of cloud cover
        nodata (tuple) : Inclusive fraction (0.0-1.0) of no data allowed in a sample

    Returns:
        df: a filtered dataframe
    """
    # temporal filtering
    if daterange is not None and 'timestamp' in df.columns:
        assert (isinstance(daterange, list) or isinstance(daterange, tuple)) and len(daterange)==2
        df = df[df.timestamp >= daterange[0]]
        df = df[df.timestamp <= daterange[1]]
    
    # spatial filtering
    if region is not None:
        idxs = df.sindex.query(region)
        df = df.take(idxs)
    # cloud filtering
    if cloud_cover is not None:
        df = df[df.cloud_cover >= cloud_cover[0]]
        df = df[df.cloud_cover <= cloud_cover[1]]

    # spatial filtering
    if nodata is not None:
        df = df[df.nodata >= nodata[0]]
        df = df[df.nodata <= nodata[1]]

    return df

def my_filter_download(df, local_dir, source_name, extract_to_dir, by_row=False, verbose=False, tif_columns=None, download_workers=DEFAULT_DOWNLOAD_WORKERS):
    """Downloads and unpacks the data of Major-TOM based on a metadata dataframe"""
    if isinstance(local_dir, str):
        local_dir = Path(local_dir)
        
    if extract_to_dir is None:
        print(" >>> No extract_to_dir provided, using local_dir")
        extract_to_dir = local_dir

    # identify all parquets that need to be downloaded (group them)
    urls = df.parquet_url.unique()
    print(f'Starting parallel download of {len(urls)} parquet files.') if verbose else None

    def process_parquet(url):
        # Create a unique temporary file for each thread
        temp_file = tempfile.NamedTemporaryFile(suffix=".parquet", dir=local_dir).name
        
        # identify all relevant rows for this parquet
        rows = df[df.parquet_url == url].parquet_row.unique()
        
        max_retries = 3
        retry_delay = 5  # seconds
        success = False
        last_error = None
        
        for attempt in range(max_retries):
            try:
                if not by_row:
                    # Create an opener with a longer timeout
                    opener = urllib.request.build_opener()
                    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
                    urllib.request.install_opener(opener)
                    
                    # Download with timeout using urlopen (30 minutes timeout)
                    with urllib.request.urlopen(url, timeout=1800) as response:
                        with open(temp_file, 'wb') as out_file:
                            out_file.write(response.read())
                    temp_path = temp_file
                else:
                    f = fsspec.open(url)
                    temp_path = f.open()
                
                # Process the downloaded parquet file
                try:
                    with pq.ParquetFile(temp_path) as pf:
                        for row_idx in rows:
                            table = pf.read_row_group(row_idx)

                            product_id = table['product_id'][0].as_py() if 'product_id' in table.column_names else "id"
                            grid_cell = table['grid_cell'][0].as_py()
                            row = grid_cell.split('_')[0]
                        
                            dest = extract_to_dir / Path(f"{source_name}/{row}/{grid_cell}/{product_id}")
                            dest.mkdir(exist_ok=True, parents=True)
                            
                            if tif_columns == 'all':
                                columns = [col for col in table.column_names if col[0] == 'B']
                                if source_name in ['Core-S2L1C', 'Core-S2L2A']:
                                    columns.append('cloud_mask')
                            elif tif_columns is None:
                                columns = []
                            else:
                                columns = tif_columns

                            # Save tifs
                            for col in columns:
                                with open(dest / f"{col}.tif", "wb") as f:
                                    f.write(table[col][0].as_py())

                            # Save thumbnail
                            with open(dest / "thumbnail.png", "wb") as f:
                                f.write(table['thumbnail'][0].as_py())

                    success = True
                    break  # Successfully processed the file, exit retry loop
                
                except Exception as e:
                    last_error = f"Error processing parquet content: {str(e)}"
                    if attempt < max_retries - 1:
                        print(f"Error processing parquet content for {url}, attempt {attempt + 1}/{max_retries}: {str(e)}")
                        time.sleep(retry_delay)
                        continue
                
                finally:
                    # Cleanup
                    if not by_row:
                        try:
                            os.remove(temp_path)
                        except:
                            pass
                    else:
                        try:
                            f.close()
                        except:
                            pass

            except urllib.error.HTTPError as e:
                last_error = f"HTTP Error {e.code}: {str(e)}"
                if e.code == 504 and attempt < max_retries - 1:
                    print(f"Timeout error for {url}, attempt {attempt + 1}/{max_retries}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries - 1:
                    print(f"Error downloading {url}, attempt {attempt + 1}/{max_retries}: {str(e)}")
                    time.sleep(retry_delay)
                    continue

        return {
            'url': url,
            'success': success,
            'error': last_error if not success else None
        }

    # Use ThreadPoolExecutor for parallel downloads
    # max_workers = min(len(urls), MAX_WORKERS*4)  # Use more workers since it's I/O bound
    max_workers = min(len(urls), download_workers)
    print(f"Using {max_workers} workers for parallel downloads") if verbose else None
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(process_parquet, url): url for url in urls}
        
        for future in tqdm(
            futures.as_completed(future_to_url),
            total=len(urls),
            desc=f'Downloading {source_name} parquets'
        ):
            results.append(future.result())

    # Process results and handle failures
    failed_downloads = [r for r in results if not r['success']]
    if failed_downloads:
        print(f"\nWarning: Failed to download {len(failed_downloads)} parquet files for {source_name}")
        print("\nFailed downloads:")
        for fail in failed_downloads:
            print(f"URL: {fail['url']}")
            print(f"Error: {fail['error']}")
            print("---")
        raise RuntimeError(f"Some parquet files failed to download for {source_name}. Please retry the download.")

    print(f"Successfully downloaded and processed {len(urls) - len(failed_downloads)} parquet files for {source_name}")


def my_metadata_from_url(access_url, local_url):
    local_url, response = urllib.request.urlretrieve(access_url, local_url)
    df = pq.read_table(local_url).to_pandas()
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df.timestamp)
    df = fix_crs(df) # Fix CRS typo if present
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.centre_lon, df.centre_lat), crs=df.crs.iloc[0]
    )
    return gdf

def get_metadata(source: str, output_dir: Path) -> gpd.GeoDataFrame:
    """Fetch metadata from HuggingFace dataset for a specific source"""
    access_url = f"https://huggingface.co/datasets/Major-TOM/{source}/resolve/main/metadata.parquet?download=true"
    local_url = output_dir / source / "metadata.parquet"
    local_url.parent.mkdir(exist_ok=True, parents=True)
    
    if local_url.exists():
        print(f"Using cached metadata for {source}")
        df = pq.read_table(local_url).to_pandas()
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df.timestamp)
        df = fix_crs(df)
        gdf = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df.centre_lon, df.centre_lat), crs=df.crs.iloc[0]
        )
    else:
        print(f"Downloading metadata for {source}...")
        gdf = my_metadata_from_url(access_url, local_url)
    
    return gdf


def filter_data(gdf, bbox, cloud_cover, date_range):
    """Filter metadata based on given parameters"""
    region = box(*bbox)
    return my_filter_metadata(
        gdf,
        cloud_cover=cloud_cover,
        region=region,
        daterange=date_range,
        nodata=(0.0, 0.0)
    )


def find_common_samples(filtered_dfs: Dict[str, gpd.GeoDataFrame]) -> Dict[str, gpd.GeoDataFrame]:
    """Find samples that share common grid cells across all datasets"""
    # Create sets of grid_cells for each dataset
    grid_cell_sets = {
        source: set(df['grid_cell'].unique())
        for source, df in filtered_dfs.items()
    }
    
    # Find intersection of all grid cell sets
    common_grid_cells = set.intersection(*grid_cell_sets.values())
    print(f"\033[92mFound {len(common_grid_cells)} common grid cells across all sources\033[0m")
    
    # Filter dataframes to keep only rows with common grid cells
    filtered_common = {}
    for source, df in filtered_dfs.items():
        filtered_common[source] = df[df['grid_cell'].isin(common_grid_cells)]
        print(f"{source}: {len(filtered_common[source])} samples for common grid cells")
    
    return filtered_common


def download_source_files(df: gpd.GeoDataFrame, output_dir: Path, source: str, extract_to_dir: Path = None, mode: str = 'full', delete_parquets: bool = False, download_workers: int = DEFAULT_DOWNLOAD_WORKERS, revalidate: bool = False):
    """Download files for a specific source"""
    print(f"Processing files for {source}...")
    
    if mode == 'download':
        # Only download parquet files without extracting
        download_parquet_files(
            df,
            local_dir=output_dir,
            source_name=source,
            download_workers=download_workers,
            revalidate=revalidate,
            verbose=True
        )
    elif mode == 'extract':
        # Extract data from already downloaded parquet files
        extract_from_parquet_files(
            df,
            local_dir=output_dir,
            source_name=source,
            extract_to_dir=extract_to_dir,
            delete_parquets=delete_parquets,
            verbose=True,
            tif_columns=CONTENT[source]
        )
    else:  # mode == 'full'
        # Use the original function for backwards compatibility
        my_filter_download(
            df,
            local_dir=output_dir,
            extract_to_dir=extract_to_dir,
            source_name=source,
            by_row=False,
            verbose=True,
            tif_columns=CONTENT[source],
            download_workers=download_workers
        )

def get_and_filter_source(args, source: str, data_dir: Path) -> gpd.GeoDataFrame:
    """Process a single source: get metadata and filter it"""
    source_dir = data_dir / source
    source_dir.mkdir(exist_ok=True, parents=True)
    
    # Get and filter metadata for each source
    gdf = get_metadata(source, data_dir)
    
    # Only apply cloud cover filter for Sentinel-2 sources
    cloud_cover_filter = tuple(args.cloud_cover) if source.startswith('Core-S2') else None
        
    filtered_df = filter_data(
        gdf,
        bbox=args.bbox,
        cloud_cover=cloud_cover_filter,
        date_range=(args.start_date, args.end_date)
    )
    print(f"Found {len(filtered_df)} samples for {source} in the specified region")
    return filtered_df

def download_source_parallel(source_df_tuple: tuple, subset_dir: Path, extract_to_dir: Path = None, mode: str = 'full', delete_parquets: bool = False, download_workers: int = DEFAULT_DOWNLOAD_WORKERS, revalidate: bool = False):
    """Download files for a source sequentially, with resume capability"""
    source, df = source_df_tuple
    source_subset_dir = subset_dir / source
    source_subset_dir.mkdir(exist_ok=True, parents=True)
    
    # Save filtered metadata
    metadata_path = source_subset_dir / "metadata.parquet"
    df.to_parquet(metadata_path)
    print(f"Saved filtered metadata for {source} to {metadata_path}")
    
    # If we're only downloading parquet files, we don't need to check for existing tif files
    if mode == 'download':
        download_source_files(df, subset_dir, source, extract_to_dir=extract_to_dir, mode=mode, delete_parquets=delete_parquets, download_workers=download_workers, revalidate=revalidate)
        print(f"Completed parquet downloads for {source}")
        return
    
    # If we're extracting and the extraction metadata exists, we don't need to check for existing files
    parquet_dir = subset_dir / source / "parquets"
    extraction_file = parquet_dir / "extraction_metadata.parquet"
    filtered_df_file = parquet_dir / "filtered_df.parquet"
    
    if mode == 'extract' and extraction_file.exists() and filtered_df_file.exists():
        print(f"Using saved extraction metadata for {source}")
        download_source_files(df, subset_dir, source, extract_to_dir=extract_to_dir, mode=mode, delete_parquets=delete_parquets, download_workers=download_workers, revalidate=revalidate)
        print(f"Completed extraction for {source}")
        return
    
    # Filter out already processed grid cells more efficiently
    def get_existing_files(df, subset_dir, source):
        # Create all possible paths
        # For DEM, use 'id' as product_id, for other sources use actual product_id
        product_ids = df['product_id'] if 'product_id' in df.columns else pd.Series(['id'] * len(df))
        grid_cells = df['grid_cell']
        row_dirs = grid_cells.str.split('_').str[0]
        
        # Vectorized path creation
        paths = [
            subset_dir / source / row_dir / grid_cell / product_id / "thumbnail.png"
            for row_dir, grid_cell, product_id in zip(row_dirs, grid_cells, product_ids)
        ]
        
        # Batch existence check
        exists_mask = [path.exists() for path in tqdm(paths, desc=f"Checking existing files for {source}", unit="file")]
        return pd.Series(exists_mask, index=df.index)
    
    # Create mask of unprocessed files
    exists_mask = get_existing_files(df, subset_dir, source)
    df_to_process = df[~exists_mask]
    
    if len(df_to_process) == 0:
        print(f"All files for {source} are already processed. Skipping.")
        return
    
    print(f"Found {len(df) - len(df_to_process)} already processed files")
    print(f"Processing remaining {len(df_to_process)} files for {source}...")
    
    # Process the remaining data files
    download_source_files(df_to_process, subset_dir, source, extract_to_dir=extract_to_dir, mode=mode, delete_parquets=delete_parquets, download_workers=download_workers, revalidate=revalidate)
    print(f"Completed processing for {source}")

def remove_duplicates(common_dfs: Dict[str, gpd.GeoDataFrame],
                      criteria: str = None) -> Dict[str, gpd.GeoDataFrame]:
    """Remove duplicates from common dataframes based on source-specific relevant columns."""
    for source, df in common_dfs.items():
        num_rows = len(df)
        
        if 'timestamp' in df.columns:
            if criteria == "latest":
                # Sort by timestamp and keep the latest
                df = df.sort_values(by='timestamp', ascending=False)
            elif criteria == None:
                raise ValueError("Please, specify a criteria for deduplication. Currently we do not support multiple timestamps for the same grid_cell.")
            else:
                raise ValueError("Criteria not supported")
        
        # TODO:
        # Product_id includes the timestamp.
        # We ignore one of the two orbit_states to avoid duplicates.
        # We can also ignore cloud_cover since we have already filtered by cloud_cover
        # We also ignore crs. Apparently, there are rows that are entirely duplicates except for the crs (? wierd)
        # We also ignore centre_lat and centre_lon since not always are aligned
        subset_columns = [col for col in df.columns if col not in [
            'parquet_row', 'parquet_url', 'geometry', 'timestamp', 'product_id',
            'orbit_state', 'cloud_cover', 'crs', 'centre_lat', 'centre_lon'
        ]]
        df = df.drop_duplicates(subset=subset_columns)
        
        # Verify no remaining duplicates in grid_cell
        if df['grid_cell'].duplicated().any():
            print(df[df['grid_cell'].duplicated()])
            raise ValueError(f"Found rows with duplicate grid_cells but different values in source {source}")
            
        common_dfs[source] = df
        print(f"\033[94mDropped {num_rows - len(df)} duplicates from {source}\033[0m")
    
    return common_dfs

def sample_common_dfs(common_dfs: Dict[str, gpd.GeoDataFrame], n_samples: int, seed: int) -> Dict[str, gpd.GeoDataFrame]:
    """Sample common dataframes to have n_samples samples per source"""
    # Get all unique grid cells that appear in all dataframes
    grid_cells_sets = [set(df['grid_cell'].unique()) for df in common_dfs.values()]
    all_grid_cells = list(set.intersection(*grid_cells_sets))
    if not all_grid_cells:
        raise ValueError("No common grid cells found across all sources")
    
    # Sort grid cells for reproducibility before sampling
    all_grid_cells.sort()
    
    # Randomly sample grid cells
    random.seed(seed)
    sampled_grid_cells = set(random.sample(all_grid_cells, min(n_samples, len(all_grid_cells))))
    
    # Filter each dataframe to only include the sampled grid cells
    result = {}
    for source, df in common_dfs.items():
        result[source] = df[df['grid_cell'].isin(sampled_grid_cells)]
        print(f"Sampled {len(result[source])} rows for {source}")
    
    return result

def is_valid_parquet(parquet_path):
    """
    Checks if a parquet file is valid and not empty.
    
    Args:
        parquet_path: Path to the parquet file
        
    Returns:
        bool: True if the parquet file is valid, False otherwise
    """
    try:
        # Check if file exists and has a non-zero size (not empty)
        if not os.path.exists(parquet_path) or os.path.getsize(parquet_path) == 0:
            return False
        
        # Try to open and read metadata from the parquet file
        with pq.ParquetFile(parquet_path) as pf:
            # Check if there's at least one row group
            if pf.num_row_groups == 0:
                return False
            
            # Try to read metadata of the first row group to verify basic integrity
            pf.metadata
            
            # Optionally, try reading a small sample of data to further verify
            table = pf.read_row_group(0, columns=['grid_cell'])
            
            return True
    except Exception as e:
        print(f"Error validating parquet file {parquet_path}: {str(e)}")
        return False

def download_parquet_files(df, local_dir, source_name, download_workers=DEFAULT_DOWNLOAD_WORKERS, revalidate=False, verbose=False):
    """Downloads only the parquet files without extracting data, saving them to disk"""
    if isinstance(local_dir, str):
        local_dir = Path(local_dir)

    # Create a directory to store parquet files
    parquet_dir = local_dir / source_name / "parquets"
    parquet_dir.mkdir(exist_ok=True, parents=True)

    # Identify all parquets that need to be downloaded
    urls = df.parquet_url.unique()
    print(f'Starting parallel download of {len(urls)} parquet files.') if verbose else None

    def download_parquet(url):
        # Get the filename from the URL
        filename = url.split('/')[-1].split('?')[0]
        parquet_path = parquet_dir / filename
        
        # Skip if file already exists and is valid (and we're not forcing revalidation)
        if parquet_path.exists() and not revalidate:
            if is_valid_parquet(parquet_path):
                return {
                    'url': url,
                    'path': parquet_path,
                    'success': True,
                    'error': None,
                    'skipped': True
                }
            else:
                # File exists but is corrupted or empty, delete it for redownload
                print(f"Found corrupted or invalid parquet file: {parquet_path}. Will redownload.")
                try:
                    os.remove(parquet_path)
                except Exception as e:
                    print(f"Warning: Failed to delete corrupted file {parquet_path}: {str(e)}")
        elif parquet_path.exists() and revalidate:
            # If we're revalidating, check the file and delete if invalid
            if not is_valid_parquet(parquet_path):
                print(f"Revalidation: Found corrupted parquet file: {parquet_path}. Will redownload.")
                try:
                    os.remove(parquet_path)
                except Exception as e:
                    print(f"Warning: Failed to delete corrupted file {parquet_path}: {str(e)}")
            else:
                # File is valid, skip download
                print(f"Revalidation: Confirmed valid parquet file: {parquet_path}")
                return {
                    'url': url,
                    'path': parquet_path,
                    'success': True,
                    'error': None,
                    'skipped': True
                }
        
        max_retries = 3
        retry_delay = 5  # seconds
        success = False
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Create an opener with a longer timeout
                opener = urllib.request.build_opener()
                opener.addheaders = [('User-agent', 'Mozilla/5.0')]
                urllib.request.install_opener(opener)
                
                # Download with timeout using urlopen (30 minutes timeout)
                with urllib.request.urlopen(url, timeout=1800) as response:
                    with open(parquet_path, 'wb') as out_file:
                        out_file.write(response.read())
                
                # Verify the downloaded file is valid
                if not is_valid_parquet(parquet_path):
                    last_error = "Downloaded file is corrupted or invalid"
                    if attempt < max_retries - 1:
                        print(f"Error: Downloaded parquet file is corrupted, attempt {attempt + 1}/{max_retries}. Retrying...")
                        os.remove(parquet_path)
                        time.sleep(retry_delay)
                        continue
                
                success = True
                break  # Successfully downloaded, exit retry loop
            
            except urllib.error.HTTPError as e:
                last_error = f"HTTP Error {e.code}: {str(e)}"
                if e.code == 504 and attempt < max_retries - 1:
                    print(f"Timeout error for {url}, attempt {attempt + 1}/{max_retries}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    continue
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries - 1:
                    print(f"Error downloading {url}, attempt {attempt + 1}/{max_retries}: {str(e)}")
                    time.sleep(retry_delay)
                    continue
                # Make sure the file is deleted if it was partially downloaded
                if parquet_path.exists():
                    try:
                        os.remove(parquet_path)
                    except:
                        pass

        return {
            'url': url,
            'path': parquet_path if success else None,
            'success': success,
            'error': last_error if not success else None,
            'skipped': False
        }

    # Use ThreadPoolExecutor for parallel downloads with the specified number of workers
    max_workers = min(len(urls), download_workers)
    print(f"Using {max_workers} workers for parallel downloads") if verbose else None
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(download_parquet, url): url for url in urls}
        
        for future in tqdm(
            futures.as_completed(future_to_url),
            total=len(urls),
            desc=f'Downloading {source_name} parquets'
        ):
            results.append(future.result())

    # Process results and handle failures
    failed_downloads = [r for r in results if not r['success']]
    skipped_downloads = [r for r in results if r['skipped']]
    
    if failed_downloads:
        print(f"\nWarning: Failed to download {len(failed_downloads)} parquet files for {source_name}")
        print("\nFailed downloads:")
        for fail in failed_downloads:
            print(f"URL: {fail['url']}")
            print(f"Error: {fail['error']}")
            print("---")
        raise RuntimeError(f"Some parquet files failed to download for {source_name}. Please retry the download.")

    print(f"Successfully downloaded {len(results) - len(failed_downloads) - len(skipped_downloads)} parquet files for {source_name}")
    print(f"Skipped {len(skipped_downloads)} valid existing parquet files")
    
    # Create a mapping from URLs to local file paths
    url_to_path = {r['url']: r['path'] for r in results if r['success']}
    
    # Save the URL to path mapping
    mapping_file = parquet_dir / "url_to_path.parquet"
    mapping_df = pd.DataFrame({
        'url': list(url_to_path.keys()),
        'path': [str(path) for path in url_to_path.values()]
    })
    mapping_df.to_parquet(mapping_file)
    
    # Save the extraction metadata - creating a dataframe that maps URLs to the rows that need to be extracted
    extraction_meta = []
    for url in urls:
        rows = df[df.parquet_url == url].parquet_row.unique()
        for row in rows:
            extraction_meta.append({
                'url': url,
                'row': row
            })
    
    extraction_df = pd.DataFrame(extraction_meta)
    extraction_file = parquet_dir / "extraction_metadata.parquet"
    extraction_df.to_parquet(extraction_file)
    
    # Also save the full filtered dataframe for reference
    filtered_df_file = parquet_dir / "filtered_df.parquet"
    df.to_parquet(filtered_df_file)
    
    print(f"Saved extraction metadata for {len(extraction_meta)} rows across {len(urls)} parquet files")
    
    return url_to_path

def extract_from_parquet_files(df, local_dir, source_name, extract_to_dir, delete_parquets=False, verbose=False, tif_columns=None):
    """Extracts data from already downloaded parquet files"""
    if isinstance(local_dir, str):
        local_dir = Path(local_dir)

    # Path to the directory where parquet files are stored
    parquet_dir = local_dir / source_name / "parquets"
    
    # Check if the URL to path mapping exists
    mapping_file = parquet_dir / "url_to_path.parquet"
    if not mapping_file.exists():
        raise FileNotFoundError(f"URL to path mapping file not found at {mapping_file}. Please download parquet files first.")
    
    # Load the URL to path mapping
    mapping_df = pd.read_parquet(mapping_file)
    url_to_path = dict(zip(mapping_df['url'], mapping_df['path']))
    
    # Try to load the extraction metadata if it exists, otherwise use the provided dataframe
    extraction_file = parquet_dir / "extraction_metadata.parquet"
    filtered_df_file = parquet_dir / "filtered_df.parquet"
    
    if extraction_file.exists() and filtered_df_file.exists():
        print("Using saved extraction metadata")
        extraction_df = pd.read_parquet(extraction_file)
        
        # We need to load the original filtered df to get all the metadata
        saved_df = pd.read_parquet(filtered_df_file)
        
        # If a specific subset of df was provided, filter extraction_df to only those URLs
        if df is not None:
            urls_to_extract = df.parquet_url.unique()
            extraction_df = extraction_df[extraction_df['url'].isin(urls_to_extract)]
            saved_df = saved_df[saved_df.parquet_url.isin(urls_to_extract)]
        
        # Replace the input df with the saved one
        df = saved_df
    else:
        # If no saved metadata, create extraction_df from the provided df
        print("No saved extraction metadata found, using provided dataframe")
        extraction_df = []
        for url in df.parquet_url.unique():
            rows = df[df.parquet_url == url].parquet_row.unique()
            for row in rows:
                extraction_df.append({
                    'url': url,
                    'row': row
                })
        extraction_df = pd.DataFrame(extraction_df)
    
    # Get all unique URLs that need to be processed
    urls = extraction_df['url'].unique()
    print(f'Starting extraction from {len(urls)} parquet files.') if verbose else None
    
    # Check if all required parquet files exist and are valid
    missing_or_invalid_urls = []
    for url in urls:
        if url not in url_to_path:
            missing_or_invalid_urls.append((url, "Missing"))
        elif not is_valid_parquet(url_to_path[url]):
            missing_or_invalid_urls.append((url, "Invalid/Corrupted"))
    
    if missing_or_invalid_urls:
        print(f"Warning: {len(missing_or_invalid_urls)} parquet files are missing or corrupted. Please download them first.")
        print("Issues with URLs:")
        for url, issue in missing_or_invalid_urls[:5]:  # Show first 5 problem URLs
            print(f"  {url} - {issue}")
        if len(missing_or_invalid_urls) > 5:
            print(f"  ... and {len(missing_or_invalid_urls) - 5} more")
        raise FileNotFoundError("Some required parquet files are missing or corrupted. Please run the download step again.")
    
    def process_parquet(url):
        # Get the local path of the parquet file
        parquet_path = url_to_path[url]
        
        # Get the rows in this parquet file that we need to extract
        rows = extraction_df[extraction_df['url'] == url]['row'].unique()
        
        success = False
        last_error = None
        
        try:
            with pq.ParquetFile(parquet_path) as pf:
                for row_idx in rows:
                    table = pf.read_row_group(row_idx)

                    product_id = table['product_id'][0].as_py() if 'product_id' in table.column_names else "id"
                    grid_cell = table['grid_cell'][0].as_py()
                    row = grid_cell.split('_')[0]
                
                    dest = extract_to_dir / Path(f"{source_name}/{row}/{grid_cell}/{product_id}")
                    dest.mkdir(exist_ok=True, parents=True)
                    
                    if tif_columns == 'all':
                        columns = [col for col in table.column_names if col[0] == 'B']
                        if source_name in ['Core-S2L1C', 'Core-S2L2A']:
                            columns.append('cloud_mask')
                    elif tif_columns is None:
                        columns = []
                    else:
                        columns = tif_columns

                    # Save tifs
                    for col in columns:
                        with open(dest / f"{col}.tif", "wb") as f:
                            f.write(table[col][0].as_py())

                    # Save thumbnail
                    with open(dest / "thumbnail.png", "wb") as f:
                        f.write(table['thumbnail'][0].as_py())
            
            success = True
            
            # Delete the parquet file if requested
            if delete_parquets:
                try:
                    os.remove(parquet_path)
                except Exception as e:
                    print(f"Warning: Failed to delete parquet file {parquet_path}: {str(e)}")
                    
        except Exception as e:
            last_error = str(e)
        
        return {
            'url': url,
            'path': parquet_path,
            'success': success,
            'error': last_error if not success else None
        }
    
    # Use ThreadPoolExecutor for parallel processing
    max_workers = min(len(urls), MAX_WORKERS)
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(process_parquet, url): url for url in urls}
        
        for future in tqdm(
            futures.as_completed(future_to_url),
            total=len(urls),
            desc=f'Extracting from {source_name} parquets'
        ):
            results.append(future.result())
    
    # Process results and handle failures
    failed_extractions = [r for r in results if not r['success']]
    if failed_extractions:
        print(f"\nWarning: Failed to extract from {len(failed_extractions)} parquet files for {source_name}")
        print("\nFailed extractions:")
        for fail in failed_extractions:
            print(f"URL: {fail['url']}")
            print(f"Path: {fail['path']}")
            print(f"Error: {fail['error']}")
            print("---")
        raise RuntimeError(f"Some parquet extractions failed for {source_name}.")
    
    print(f"Successfully extracted data from {len(results) - len(failed_extractions)} parquet files for {source_name}")
    
    # Clean up the metadata files if all parquet files were deleted
    if delete_parquets and not any(os.path.exists(r['path']) for r in results):
        try:
            # Delete all metadata files
            for meta_file in [mapping_file, extraction_file, filtered_df_file]:
                if meta_file.exists():
                    os.remove(meta_file)
            
            # Try to remove the parquets directory if it's empty
            if os.path.exists(parquet_dir) and not os.listdir(parquet_dir):
                os.rmdir(parquet_dir)
                
            print(f"Cleaned up metadata files and directory for {source_name}")
        except Exception as e:
            print(f"Warning: Failed to clean up metadata files or directory: {str(e)}")
    
    return results

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    
    data_dir = Path(args.data_dir)
    subset_dir = data_dir / args.subset_name

    if args.extract_to_dir is None:
        print(" >>> No extract_to_dir provided, using local_dir")
        extract_to_dir = data_dir / args.subset_name
    else:
        extract_to_dir = Path(args.extract_to_dir) / args.subset_name
    
    # Always process metadata and filtering for all modes
    print("\033[92mFetching and filtering metadata...\033[0m")
    
    # Parallel processing of metadata fetching and filtering
    max_workers = min(len(args.sources), MAX_WORKERS)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_source = {
            executor.submit(get_and_filter_source, args, source, data_dir): source 
            for source in args.sources
        }
        
        # Collect results while maintaining order
        filtered_dfs = {}
        for future in futures.as_completed(future_to_source):
            source = future_to_source[future]
            try:
                filtered_dfs[source] = future.result()
            except Exception as e:
                print(f"Error processing {source}: {e}")
                raise e

    # Synchronization point: find common samples across all sources
    common_dfs = find_common_samples(filtered_dfs)
    
    # Remove duplicates for each of the common_dfs
    common_dfs = remove_duplicates(common_dfs, criteria=args.criteria)
    
    # After removing duplicates, print the number of samples for each source
    print("\033[92mAfter removing duplicates:\033[0m")
    for source, df in common_dfs.items():
        print(f"{source}: {len(df)} samples for common grid cells")
    
    if args.preview:
        return
    
    if args.n_samples is not None: # Else, we download all samples.
        print(f"Sampling {args.n_samples} samples per source...")
        common_dfs = sample_common_dfs(common_dfs, args.n_samples, args.seed)
        print(f"Done sampling {args.n_samples} grid cells per source!")

    # Remove modalities that we don't want to process (after the sampling / matching)
    for mod in args.sources:
        if mod not in args.modalities:
            common_dfs.pop(mod)
    
    # common_dfs.pop('Core-DEM')
    # common_dfs.pop('Core-S1RTC')
    # common_dfs.pop('Core-S2L1C')
    # common_dfs.pop('Core-S2L2A')
    print(f"We will only process the following modalities: {list(common_dfs.keys())}")
    
    # Print information about download workers
    if args.mode in ['download', 'full']:
        print(f"\033[94mUsing {args.download_workers} workers for parallel downloads\033[0m")
        print("If downloads are slow, try reducing this number with the --download-workers parameter")
        
        if args.revalidate:
            print("\033[94mRevalidating all parquet files (will check for corrupted files)\033[0m")
        else:
            print("Use --revalidate to force checking of existing parquet files for corruption")
    
    # Execute the appropriate action based on mode
    if args.mode == 'download':
        print("\033[92mStarting download of parquet files...\033[0m")
        for source, df in common_dfs.items():
            print(f"\033[94mDownloading parquets for modality: {source}\033[0m")
            download_source_parallel((source, df), subset_dir, mode='download', 
                                    extract_to_dir=extract_to_dir,
                                    delete_parquets=args.delete_parquets, 
                                    download_workers=args.download_workers,
                                    revalidate=args.revalidate)
        print("\033[92mParquet file download complete.\033[0m")
        print("To extract data from these parquet files, run this script with --mode extract")

    elif args.mode == 'extract':
        print("\033[92mStarting extraction from parquet files...\033[0m")
        for source, df in common_dfs.items():
            print(f"\033[94mExtracting data for modality: {source}\033[0m")
            download_source_parallel((source, df), subset_dir, mode='extract', 
                                    extract_to_dir=extract_to_dir,
                                    delete_parquets=args.delete_parquets,
                                    download_workers=args.download_workers,
                                    revalidate=args.revalidate)
        print("\033[92mData extraction complete.\033[0m")
        if args.delete_parquets:
            print("Parquet files have been deleted.")
        else:
            print("To delete the parquet files, run this script with --mode extract --delete-parquets")

    else:  # mode == 'full'
        print("\033[92mStarting full download and extraction process...\033[0m")
        for source, df in common_dfs.items():
            print(f"\033[94mProcessing modality: {source}\033[0m")
            download_source_parallel((source, df), subset_dir, mode='full',
                                    extract_to_dir=extract_to_dir,
                                    download_workers=args.download_workers,
                                    revalidate=args.revalidate)
        print("\033[92mDownload and extraction complete.\033[0m")

if __name__ == "__main__":
    main()