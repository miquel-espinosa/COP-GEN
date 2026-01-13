from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import rasterio

import logging


def read_tif(path: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
    with rasterio.open(path) as src:
        arr = src.read()
        meta = src.meta.copy()
    return arr, meta


def default_meta_if_missing() -> Dict[str, Any]:
    return {
        'driver': 'GTiff',
        'width': 1,
        'height': 1,
        'crs': 'EPSG:4326',
        'transform': rasterio.transform.from_bounds(-180, -90, 180, 90, 1, 1),
    }


def write_tif(
    path: Union[str, Path],
    data: np.ndarray,
    ref_meta_or_path: Optional[Union[Dict[str, Any], str, Path]] = None,
    dtype: Optional[np.dtype] = None,
    nodata: Optional[Union[int, float]] = None,
) -> None:
    """
    Save data to a GeoTIFF, using meta from ref or a default world-CRS 1x1 template.
    Ensures band dimension is first; adds nodata if provided.
    """
    path = Path(path)

    if data.ndim == 2:
        data = data[np.newaxis, ...]

    if ref_meta_or_path is None:
        logging.warning("Reference metadata is None, using default metadata")
        meta = default_meta_if_missing()
    elif isinstance(ref_meta_or_path, (str, Path)):
        with rasterio.open(ref_meta_or_path) as src:
            meta = src.meta.copy()
    else:
        meta = dict(ref_meta_or_path)

    count = data.shape[0]
    meta.update({
        'count': count,
        'height': data.shape[-2],
        'width': data.shape[-1],
    })
    if dtype is not None:
        meta['dtype'] = dtype
    if nodata is not None:
        meta['nodata'] = nodata

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, 'w', **meta) as dst:
        dst.write(data)


