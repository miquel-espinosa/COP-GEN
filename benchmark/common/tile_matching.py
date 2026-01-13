from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence


DEFAULT_PATTERNS: Sequence[str] = (
    "{tile}.tif",
    "{tile}_*.tif",
)


def find_matching_file(directory: Path, tile_name: str, patterns: Sequence[str] = DEFAULT_PATTERNS) -> Optional[Path]:
    """
    Find a file in `directory` matching a tile name according to allowed patterns.
    Returns the first match or None.
    """
    for pattern in patterns:
        matches = list(Path(directory).glob(pattern.format(tile=tile_name)))
        if matches:
            return matches[0]
    return None


