import os
import argparse
from pathlib import Path
from typing import List, Dict, Iterable, Optional

from PIL import Image  # type: ignore


# -----------------------------------------------------------------------------
# Configuration – edit these as needed or pass via command-line arguments
# -----------------------------------------------------------------------------
DEFAULT_MODALITY_TO_FILENAME: Dict[str, str] = {
    "3d_cartesian_lat_lon": "band_0.png",
    # "DEM_DEM": "band_0.png",
    "DEM_DEM": "band_0_gray.png",
    "LULC_LULC": "band_0.png",
    "mean_timestamps": "band_0.png",
    "S1RTC_vh_vv": "composite_s1rtc.png",
    "S2L1C_cloud_mask": "band_0.png",
    "S2L1C_B01_B09_B10": "composite_0_1_2.png",
    "S2L1C_B02_B03_B04_B08": "composite_0_1_2.png",
    "S2L1C_B05_B06_B07_B11_B12_B8A": "composite_0_1_2.png",
    "S2L2A_B01_B09": "composite_remaining.png",
    "S2L2A_B02_B03_B04_B08": "composite_0_1_2.png",
    "S2L2A_B05_B06_B07_B11_B12_B8A": "composite_0_1_2.png",
}


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def discover_visualisation_numbers(results_path: Path, modalities: List[str]) -> List[int]:
    """Return all visualisation indices present in any modality folder.

    Looks for sub-folders named exactly ``visualisations-<num>``.
    """
    numbers = set()
    for modality in modalities:
        mod_dir = results_path / modality
        if not mod_dir.is_dir():
            continue
        for child in mod_dir.iterdir():
            if not child.is_dir():
                continue
            if not child.name.startswith("visualisations-"):
                continue
            try:
                numbers.add(int(child.name.split("-")[-1]))
            except ValueError:
                continue
    return sorted(numbers)


def build_image_paths(results_path: Path, modalities: Iterable[str], subfolder: str,
                      modality_to_filename: Dict[str, str]) -> List[Path]:
    """Return expected image paths for each modality within the given subfolder."""
    paths = []
    for modality in modalities:
        filename = modality_to_filename.get(modality, "merged.png")
        path = results_path / modality / subfolder / filename
        if not path.exists():
            path = results_path / modality / subfolder / "merged.png"
        paths.append(path)
    return paths


def merge_and_save(image_paths: List[Path], out_path: Path, overwrite: bool, verbose: bool) -> bool:
    """Load images, merge vertically, and save.

    Returns True if a file was written (or skipped due to existing when not overwriting),
    False if there were no images to merge.
    """
    if out_path.exists() and not overwrite:
        if verbose:
            print(f"⏭️  Skipping existing {out_path}")
        return True

    imgs = load_and_rescale_images(image_paths)
    if not imgs:
        if verbose:
            print("⚠️  No images found for requested merge")
        return False

    merged = stack_vertically(imgs)
    merged.save(out_path)
    print(f"✅ Saved → {out_path}")
    return True


def load_and_rescale_images(paths: List[Path]) -> List[Image.Image]:
    """Open all images and upscale them so they share the maximum width.

    Heights are scaled proportionally to avoid stretching the images.
    Missing paths are ignored (with a warning).
    """
    images = []
    for p in paths:
        if not p.exists():
            print(f"⚠️  Missing image - skipping: {p}")
            continue
        try:
            images.append(Image.open(p).convert("RGB"))
        except Exception as exc:
            print(f"⚠️  Could not open {p}: {exc}")
    if not images:
        return []

    max_width = max(img.width for img in images)
    scaled: List[Image.Image] = []
    for img in images:
        if img.width == max_width:
            scaled.append(img)
            continue
        new_height = int(img.height * max_width / img.width)
        scaled.append(img.resize((max_width, new_height), Image.LANCZOS))
    return scaled


def stack_vertically(images: List[Image.Image]) -> Image.Image:
    """Return a single image created by stacking *images* top to bottom."""
    if not images:
        raise ValueError("No images provided for stacking")
    total_height = sum(img.height for img in images)
    max_width = max(img.width for img in images)
    merged = Image.new("RGB", (max_width, total_height))
    y = 0
    for img in images:
        x = (max_width - img.width) // 2  # centre each image horizontally
        merged.paste(img, (x, y))
        y += img.height
    return merged


# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge per-modality visualisations by stacking the images vertically.")
    parser.add_argument("results_path", type=Path,
                        help="Root directory containing per-modality sub-folders")
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Directory to store merged images (default: <results_path>/merged_visualisations)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing merged images if they already exist")
    return parser.parse_args()


def merge_visualisations(results_path: str, output_dir: str = None, overwrite: bool = False, verbose: bool = True):
    """Merge per-modality visualisations by stacking the images vertically."""
    
    modality_to_filename = DEFAULT_MODALITY_TO_FILENAME
    modalities = list(modality_to_filename.keys())

    results_path: Path = Path(results_path).expanduser().resolve()
    output_dir: Path = Path(output_dir or results_path / "merged_visualisations").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"📂 Results path  : {results_path}")
        print(f"📂 Output dir    : {output_dir}")
        print(f"📋 Modalities    : {modalities}\n")

    numbers = discover_visualisation_numbers(results_path, modalities)
    if not numbers:
        if any((results_path / m / "visualisations").is_dir() for m in modalities):
            image_paths = build_image_paths(results_path, modalities, "visualisations", modality_to_filename)
            merged_path = output_dir / "merged.png"
            if merge_and_save(image_paths, merged_path, overwrite, verbose):
                return
        print("❌ No visualisation folders found - exiting.")
        return

    if verbose:
        print(f"🔢 Found visualisations: {numbers}\n")

    for num in numbers:
        subfolder = f"visualisations-{num}"
        image_paths = build_image_paths(results_path, modalities, subfolder, modality_to_filename)
        merged_path = output_dir / f"merged-{num}.png"
        if not merge_and_save(image_paths, merged_path, overwrite, verbose):
            if verbose:
                print(f"⚠️  No images found for visualisation {num} - skipping")
        

def main():
    args = parse_args()
    
    merge_visualisations(results_path=args.results_path, output_dir=args.output_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main() 