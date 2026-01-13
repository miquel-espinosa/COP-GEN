from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Union
import numpy as np


# Central place to define modality-related constants and helpers

VALID_MODALITIES: List[str] = [
    "DEM",
    "LULC",
    "S1RTC",
    "S2L2A",
    "S2L1C",
    "coords",
]


S2_BAND_ORDER: Dict[str, List[str]] = {
    "S2L2A": [
        "B01.tif",
        "B02.tif",
        "B03.tif",
        "B04.tif",
        "B05.tif",
        "B06.tif",
        "B07.tif",
        "B08.tif",
        "B8A.tif",
        "B09.tif",
        "B11.tif",
        "B12.tif",
    ],
    "S2L1C": [
        "B01.tif",
        "B02.tif",
        "B03.tif",
        "B04.tif",
        "B05.tif",
        "B06.tif",
        "B07.tif",
        "B08.tif",
        "B8A.tif",
        "B09.tif",
        "B10.tif",
        "B11.tif",
        "B12.tif",
    ],
}

def get_output_dtype_for_modality(modality: str) -> np.dtype:
    """
    Return numpy dtype for saved GeoTIFF based on modality.
    Mirrors the logic present in generator save routine.
    """
    if modality in ("S2L2A", "S2L1C"):
        return np.uint16
    if modality == "LULC":
        return np.uint8
    return np.float32


def get_nodata_for_dtype(dtype: np.dtype) -> Optional[Union[int, float]]:
    """Standard nodata sentinel per dtype, matching existing behavior."""
    return {np.uint8: 255, np.uint16: 65535, np.float32: -9999.0}.get(dtype, None)


