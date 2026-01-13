import argparse
import os
from pathlib import Path
from typing import List, Dict

from ddm.utils import construct_class_by_name

DEFAULT_MODALITY_DIR_CANDIDATES: Dict[str, List[str]] = {
    "DEM": ["Core-DEM", "Core-DEM-tar-gz"],
    "LULC": ["Core-LULC", "Core-LULC-tar-gz"],
    "S1RTC": ["Core-S1RTC", "Core-S1RTC-tar-gz"],
    "S2L2A": ["Core-S2L2A", "Core-S2L2A-tar-gz"],
    "S2L1C": ["Core-S2L1C", "Core-S2L1C-tar-gz"],
    "GOOGLE_EMBEDS": ["Core-GOOGLE_EMBEDS"],
}

DEFAULT_BANDS_BY_MODALITY: Dict[str, List[str]] = {
    "S1RTC": ["vv", "vh"],
    "S2L2A": ["B04", "B03", "B02", "B08"],
    "S2L1C": ["B04", "B03", "B02", "B08"],
    "DEM": ["DEM"],
}


def find_modality_dir(root_dir: Path, modality: str) -> Path | None:
    for candidate in DEFAULT_MODALITY_DIR_CANDIDATES.get(modality, []):
        p = root_dir / candidate
        if p.exists():
            return p
    return None


def ensure_cache_for_modality(modality_dir: Path, modality: str, tif_bands: List[str] | None = None) -> None:
    if tif_bands is None:
        tif_bands = DEFAULT_BANDS_BY_MODALITY.get(modality, ["B04", "B03", "B02", "B08"])
    cfg = dict(
        class_name='ddm.data.SatelliteDataset',
        data_dir=str(modality_dir),
        tif_bands=tif_bands,
        augment_horizontal_flip=False,
        center_crop=False,
        normalize_to_neg_one_to_one=False,
        satellite_type=modality,
        preprocess_bands=False,
        patchify=False,
    )
    # Instantiating triggers discovery and cache write if missing
    _ = construct_class_by_name(**cfg)


def main():
    parser = argparse.ArgumentParser("Precompute .cache_<sat>.pkl files under a world root")
    parser.add_argument("--root_dir", type=str, required=True, help="Path to folder containing Core-<modality> subfolders")
    parser.add_argument("--modalities", type=str, nargs='+', required=True, help="Modalities to process")
    args = parser.parse_args()

    root_dir = Path(args.root_dir or os.environ.get("MAJORTOM_root_dir", ".")).resolve()
    if not root_dir.exists():
        raise FileNotFoundError(f"World root not found: {root_dir}")

    for modality in args.modalities:
        if modality == 'GOOGLE_EMBEDS':
            # Skip by default unless explicitly requested
            pass
        mdir = find_modality_dir(root_dir, modality)
        if mdir is None:
            print(f"[skip] Could not find directory for {modality} under {root_dir}")
            continue
        print(f"[create] Ensuring cache for {modality} at {mdir}")
        ensure_cache_for_modality(mdir, modality)
        cache_path = mdir / f".cache_{modality}.pkl"
        if cache_path.exists():
            print(f"[ok] {cache_path}")
        else:
            print(f"[warn] Cache expected but not found: {cache_path}")


if __name__ == "__main__":
    main()
