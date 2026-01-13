import os
import tarfile
import argparse
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


def create_output_dir(output_path):
    os.makedirs(output_path, exist_ok=True)


def tar_gz_folder(folder_path, output_path):
    folder_path = Path(folder_path)
    tar_file_name = output_path / f"{folder_path.name}.tar.gz"
    with tarfile.open(tar_file_name, "w:gz") as tar:
        tar.add(folder_path, arcname=folder_path.name)
    return str(tar_file_name)


def copy_metadata_parquet(source_path, output_path):
    source_path = Path(source_path)
    output_path = Path(output_path)
    metadata_file = source_path / "metadata.parquet"
    if metadata_file.exists():
        shutil.copy(metadata_file, output_path)
        print(f"\033[92mCopied: {metadata_file} to {output_path}\033[0m")
    else:
        print(f"\033[91mNo metadata.parquet found in {source_path}\033[0m")


def process_all_folders(source_path, output_path, num_workers):
    source_path = Path(source_path)
    output_path = Path(output_path)
    create_output_dir(output_path)

    subfolders = [f for f in source_path.iterdir() if f.is_dir()]
    total_folders = len(subfolders)
    print(f"Total folders to process: {total_folders}")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(tar_gz_folder, folder, output_path)
            for folder in subfolders
        ]
        for future in tqdm(futures, desc="Processing folders", total=total_folders):
            try:
                result = future.result()
                # print(f"Created: {result}")
            except Exception as e:
                print(f"Error: {e}")

    copy_metadata_parquet(source_path, output_path)


def main():
    parser = argparse.ArgumentParser(description="Tar and gzip subfolders in parallel.")
    parser.add_argument("source", type=str, help="Path to the source directory")
    parser.add_argument("destination", type=str, help="Path to the destination directory")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers (default: 4)")

    args = parser.parse_args()
    process_all_folders(args.source, args.destination, args.workers)


if __name__ == "__main__":
    main()
