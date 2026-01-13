import os
import subprocess
import logging
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import shutil

def parse_arguments():
    parser = argparse.ArgumentParser(description="Efficiently copy and extract tar.gz files in parallel.")
    parser.add_argument("--source-dir", required=True, type=Path,
                        help="Source directory containing .tar.gz files")
    parser.add_argument("--tmp-tars-dir", required=True, type=Path,
                        help="Temporary directory for tar.gz copies")
    parser.add_argument("--dest-dir", required=True, type=Path,
                        help="Destination directory for extracted content")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of parallel processes (default: 8)")
    return parser.parse_args()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger("").addHandler(console)

def copy_extract_delete(file_path: Path, tmp_tars_dir: Path, dest_dir: Path, index: int, total: int):
    filename = file_path.name
    tmp_tar_path = tmp_tars_dir / filename

    try:
        logging.info(f"[{index}/{total}] 🧾 Copying {filename} to {tmp_tar_path} ...")

        # Use reflink copy for efficiency, fallback to standard cp
        try:
            subprocess.run(["cp", "--reflink=auto", str(file_path), str(tmp_tar_path)], check=True)
        except subprocess.CalledProcessError:
            # Fallback to shutil
            logging.info(f"[{index}/{total}] ❌ Failed to copy {filename} to {tmp_tar_path} using reflink, falling back to shutil")
            shutil.copy2(file_path, tmp_tar_path)

        logging.info(f"[{index}/{total}] 📦 Extracting {filename} ...")
        dest_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["tar", "--use-compress-program=pigz", "-xf", str(tmp_tar_path), "-C", str(dest_dir)],
            check=True
        )

        logging.info(f"[{index}/{total}] 🧹 Removing {filename} from {tmp_tar_path} ...")
        tmp_tar_path.unlink()

        logging.info(f"[{index}/{total}] ✅ Completed {filename}")
        return f"Done: {filename}"

    except Exception as e:
        logging.error(f"[{index}/{total}] ❌❌❌ Error with {filename}: {e}")
        return f"Error: {filename}"

def main():
    args = parse_arguments()

    # Ensure directories exist
    args.tmp_tars_dir.mkdir(parents=True, exist_ok=True)
    args.dest_dir.mkdir(parents=True, exist_ok=True)

    tar_files = sorted(args.source_dir.glob("*.tar.gz"))
    total = len(tar_files)
    logging.info(f"Found {total} tar.gz files to process.")

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(copy_extract_delete, f, args.tmp_tars_dir, args.dest_dir, i + 1, total): f
            for i, f in enumerate(tar_files)
        }

        for future in as_completed(futures):
            result = future.result()
            logging.info(result)

if __name__ == "__main__":
    main()