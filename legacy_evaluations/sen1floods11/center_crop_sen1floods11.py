import argparse
import os
from pathlib import Path

import rasterio
from rasterio.windows import Window


DEF_ROOT = "./data/sen1floods11_v1.1/v1.1"
IN_REL = "data/flood_events/HandLabeled"


def center_crop_tif(input_path: Path, output_path: Path, crop_size: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(input_path) as src:
        height, width = src.height, src.width
        if height < crop_size or width < crop_size:
            print(f"[WARN] Skipping (smaller than crop): {input_path} ({width}x{height}) < {crop_size}")
            return

        row_off = (height - crop_size) // 2
        col_off = (width - crop_size) // 2

        window = Window(col_off=col_off, row_off=row_off, width=crop_size, height=crop_size)
        data = src.read(window=window)
        transform = rasterio.windows.transform(window, src.transform)

        profile = src.profile.copy()
        profile.update({
            "height": crop_size,
            "width": crop_size,
            "transform": transform,
            "count": data.shape[0],
        })

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data)


def process_directory(src_dir: Path, dst_dir: Path, crop_size: int) -> None:
    tif_patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
    files = []
    for pattern in tif_patterns:
        files.extend(src_dir.rglob(pattern))

    if not files:
        print(f"[INFO] No TIFF files found in {src_dir}")
        return

    print(f"[INFO] Cropping {len(files)} files from {src_dir} -> {dst_dir} (size={crop_size})")
    for f in files:
        rel = f.relative_to(src_dir)
        out_path = dst_dir / rel
        center_crop_tif(f, out_path, crop_size)


def main():
    parser = argparse.ArgumentParser(description="Center-crop Sen1Floods11 S1/S2/Label GeoTIFFs.")
    parser.add_argument("--size", type=int, required=True, help="Crop size (e.g., 192)")
    parser.add_argument("--src-root", type=str, default=os.path.join(DEF_ROOT, IN_REL),
                        help="Input root (default: v1.1/data/flood_events/HandLabeled)")
    parser.add_argument("--dst-root", type=str, default=None,
                        help="Output root (default: v1.1/data_{size}/flood_events/HandLabeled)")
    args = parser.parse_args()

    src_root = Path(args.src_root)
    if args.dst_root is None:
        dst_root = Path(DEF_ROOT) / f"data_{args.size}" / "flood_events" / "HandLabeled"
    else:
        dst_root = Path(args.dst_root)

    subdirs = ["S1Hand", "S2Hand", "LabelHand"]
    for sub in subdirs:
        src_dir = src_root / sub
        dst_dir = dst_root / sub
        if not src_dir.exists():
            print(f"[WARN] Missing source directory: {src_dir} (skipping)")
            continue
        process_directory(src_dir, dst_dir, args.size)

    print("[DONE] Center-cropping complete.")


if __name__ == "__main__":
    main()
