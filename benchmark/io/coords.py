from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import json
import torch


def load_coords_json(path: Path) -> Dict[str, str]:
    with open(path, 'r') as f:
        return json.load(f)


def parse_coord_str(coord_str: str) -> Tuple[float, float]:
    # Accept formats like "lat=-16.50 lon=-63.50"
    parts = coord_str.replace('lat=', '').replace('lon=', '').split()
    if len(parts) != 2:
        raise ValueError(f"Invalid coordinate string: {coord_str}")
    lat, lon = float(parts[0]), float(parts[1])
    return lat, lon


def coords_tensor_for_tile(tile_name: str, coords_data: Dict[str, str], device: torch.device) -> torch.Tensor:
    if tile_name not in coords_data:
        raise KeyError(f"Coordinates not found for tile {tile_name}")
    lat, lon = parse_coord_str(coords_data[tile_name])
    # Shape [1, 2] with [lon, lat] order as expected by tokenizer
    return torch.tensor([[lon, lat]], dtype=torch.float32, device=device)


