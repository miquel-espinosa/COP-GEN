from __future__ import annotations

from pathlib import Path
from typing import Dict, List


def data_input_dir(dataset_root: Path, modality: str) -> Path:
    return Path(dataset_root) / "terramind_inputs" / modality


def data_output_dir(dataset_root: Path, experiment_name: str) -> Path:
    return Path(dataset_root) / "outputs" / "terramind" / experiment_name / "generations"


def coords_json_path(dataset_root: Path) -> Path:
    return Path(dataset_root) / "terramind_inputs" / "coords" / "tile_to_coords.json"


def logs_day_dir(dataset_root: Path, experiment_name: str, date_str: str) -> Path:
    return Path(dataset_root) / "outputs" / "terramind" / experiment_name / "logs" / date_str


def comparisons_vis_dir(dataset_root: Path, experiment_name: str) -> Path:
    return Path(dataset_root) / "outputs" / "terramind" / experiment_name / "visualisations"


def singles_vis_dir(dataset_root: Path, in_or_out: str, modality_path: str) -> Path:
    return Path(dataset_root) / "outputs" / "terramind" / "singles" / in_or_out / modality_path


