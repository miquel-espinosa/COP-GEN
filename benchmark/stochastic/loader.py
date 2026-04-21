"""
HuggingFace-based loader for the COP-GEN stochastic benchmark.

Streams the ``Major-TOM/COP-GEN-Benchmark`` dataset from HuggingFace and,
for each cell, returns paired tensors ready for metric computation:

    (real, copgen, terramind), each of shape (16, 12, 192, 192) float32

* ``real``      — 16 cloud-free Sentinel-2 L2A acquisitions
* ``copgen``    — 16 COP-GEN samples (subsampled from 33 via
                  :mod:`benchmark.seeds`)
* ``terramind`` — 16 TerraMind samples

All tensors are normalised to approximately ``[0, 1]`` by dividing by
10000 (standard S2 reflectance scaling); COP-GEN samples can slightly
exceed this range.

The 192x192 evaluation footprint is extracted from the published 1056x1056
tiles with :func:`benchmark.footprint.crop_benchmark_footprint`.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import rasterio

from .footprint import crop_benchmark_footprint, load_grid
from .seeds import load_selection

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_DATASET = "Major-TOM/COP-GEN-Benchmark"
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
         "B08", "B8A", "B09", "B11", "B12"]
MODEL_PX = 192
S2_SCALE = 10000.0
MIN_CENTRE_VALID = 0.95  # drop real acquisitions with >5% nodata in centre


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class CellSample:
    """All benchmark tensors for a single cell."""
    cell_id: str
    real: np.ndarray          # (N, 12, 192, 192) float32, N <= 16
    copgen: np.ndarray        # (16, 12, 192, 192) float32
    terramind: np.ndarray     # (16, 12, 192, 192) float32
    real_dates: List[str]
    copgen_seeds: List[int]
    terramind_seeds: List[int]


# ---------------------------------------------------------------------------
# Band decoding
# ---------------------------------------------------------------------------

def _decode_band(blob: bytes, cell_id: str, grid: Dict) -> np.ndarray:
    """Decode a per-band GeoTIFF blob and extract the 192x192 evaluation
    window via the documented centre crop."""
    with rasterio.open(io.BytesIO(blob)) as src:
        return crop_benchmark_footprint(src, cell_id, grid)[0]  # drop band dim


def _row_to_tensor(row: Dict, grid: Dict) -> np.ndarray:
    """Decode all 12 bands of a dataset row into a (12, 192, 192) array."""
    cell_id = row["grid_cell"]
    stacks = [_decode_band(row[b], cell_id, grid) for b in BANDS]
    return np.stack(stacks, axis=0).astype(np.float32) / S2_SCALE




# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

class BenchmarkLoader:
    """Per-cell access to the HF benchmark configs.

    Builds a lightweight ``cell_id -> (shard_file, row_within_shard)``
    index from each parquet shard's ``grid_cell`` column.  Each cell's
    tensors are fetched on demand via a single parquet read.

    When ``max_cells`` is set, indexing stops as soon as the three
    configs have covered that many cells in common, so smoke tests do
    not have to download the full 72\u00a0GB dataset.
    """

    def __init__(
        self,
        hf_dataset: str = HF_DATASET,
        cache_dir: Optional[Path] = None,
        max_cells: Optional[int] = None,
        local_root: Optional[Path] = None,
    ):
        self.hf_dataset = hf_dataset
        self.cache_dir = cache_dir
        self.max_cells = max_cells
        self.local_root = Path(local_root) if local_root is not None else None

        if self.local_root is not None:
            log.info("Using local root %s (HF download skipped)", self.local_root)
            grid_path = self.local_root / "metadata" / "benchmark_grid.json"
            self._shards: Dict[str, List[str]] = {
                config: sorted(str(p) for p in (self.local_root / config / "data").glob("*.parquet"))
                for config in ("real", "copgen", "terramind")
            }
        else:
            from huggingface_hub import hf_hub_download, list_repo_files

            log.info("Preparing %s (lazy index) ...", hf_dataset)

            # Download the benchmark_grid.json (needed by crop_benchmark_footprint)
            grid_path = hf_hub_download(
                repo_id=hf_dataset,
                filename="metadata/benchmark_grid.json",
                repo_type="dataset",
                cache_dir=str(cache_dir) if cache_dir else None,
            )

            # Discover shard paths
            repo_files = list_repo_files(hf_dataset, repo_type="dataset")
            self._shards = {
                config: sorted(f for f in repo_files
                               if f.startswith(f"{config}/data/") and f.endswith(".parquet"))
                for config in ("real", "copgen", "terramind")
            }

        self.grid = load_grid(grid_path)

        # Load canonical seed selection (shipped with this package)
        self.seed_selection = load_selection()

        # Build indices per config, with early-stop based on max_cells
        log.info("Indexing shards by cell (early-stop after %s cells)...",
                 max_cells or "no limit")
        self._index: Dict[str, Dict[str, List[Tuple[str, int]]]] = {}
        for config in ("real", "copgen", "terramind"):
            self._index[config] = self._build_cell_index(config)

        total = sum(len(v) for v in self._index["real"].values())
        log.info("Indexed %d real rows across %d cells",
                 total, len(self._index["real"]))

    def _fetch_shard(self, rel_path: str, retry: int = 2) -> str:
        """Return the local path to a shard.

        With ``local_root`` set this is a no-op; otherwise downloads from
        HuggingFace with retry on consistency-check failures."""
        if self.local_root is not None:
            return rel_path  # already absolute local path

        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import HfHubHTTPError

        last_err: Optional[Exception] = None
        for attempt in range(retry + 1):
            try:
                return hf_hub_download(
                    repo_id=self.hf_dataset,
                    filename=rel_path,
                    repo_type="dataset",
                    cache_dir=str(self.cache_dir) if self.cache_dir else None,
                    force_download=(attempt > 0),
                )
            except (OSError, HfHubHTTPError) as e:
                log.warning("%s attempt %d/%d failed: %s",
                            rel_path, attempt + 1, retry + 1, e)
                last_err = e
        raise RuntimeError(f"Failed to download {rel_path} after {retry + 1} attempts: {last_err}")

    def _build_cell_index(self, config: str) -> Dict[str, List[Tuple[str, int]]]:
        """For a config, return ``{cell_id: [(shard_path, row_in_shard), ...]}``.

        Downloads shards sequentially, reading only the ``grid_cell`` column
        from each.  If ``self.max_cells`` is set, stops after enough distinct
        cells are indexed so smoke tests don't download the full dataset.
        """
        import pyarrow.parquet as pq
        idx: Dict[str, List[Tuple[str, int]]] = {}
        for rel in self._shards[config]:
            if self.max_cells is not None and len(idx) >= self.max_cells:
                log.info("  %s: stopping at %d cells (>= max_cells=%d)",
                         config, len(idx), self.max_cells)
                break
            local = self._fetch_shard(rel)
            col = pq.read_table(local, columns=["grid_cell"]).column("grid_cell").to_pylist()
            for i, cell in enumerate(col):
                idx.setdefault(cell, []).append((local, i))
        return idx

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------

    def list_cells(self) -> List[str]:
        """Cells available across all three configs."""
        real_cells = set(self._index["real"])
        cg_cells = set(self._index["copgen"])
        tm_cells = set(self._index["terramind"])
        return sorted(real_cells & cg_cells & tm_cells)

    def load_cell(self, cell_id: str) -> Optional[CellSample]:
        """Fully load one cell. Returns ``None`` if <2 real samples pass
        the centre-validity filter."""
        real_stack, real_dates = self._load_config("real", cell_id, filter_valid=True)
        if len(real_stack) < 2:
            return None

        wanted = set(self.seed_selection.get(cell_id, []))
        cg_stack, cg_seeds = self._load_config(
            "copgen", cell_id, seed_filter=wanted, sort_by_seed=True,
        )
        tm_stack, tm_seeds = self._load_config(
            "terramind", cell_id, sort_by_seed=True,
        )

        return CellSample(
            cell_id=cell_id,
            real=real_stack,
            copgen=cg_stack,
            terramind=tm_stack,
            real_dates=real_dates,
            copgen_seeds=cg_seeds,
            terramind_seeds=tm_seeds,
        )

    # ---------------------------------------------------------------
    # Unified per-config loader
    # ---------------------------------------------------------------

    def _load_config(
        self,
        config: str,
        cell_id: str,
        seed_filter: Optional[set] = None,
        sort_by_seed: bool = False,
        filter_valid: bool = False,
    ) -> Tuple[np.ndarray, list]:
        """Load all rows for a cell in a given config, return (stack, meta).

        ``meta`` is a list of dates (for real) or seed ints (for models).
        Reads are batched by shard to avoid re-opening parquet files.
        """
        import pyarrow.parquet as pq

        pointers = self._index[config].get(cell_id, [])
        if not pointers:
            return np.zeros((0, 12, MODEL_PX, MODEL_PX), dtype=np.float32), []

        cols_needed = ["grid_cell", "sample_id", "date"] + BANDS

        # Group row indices by shard for batched reads
        by_shard: Dict[str, List[int]] = {}
        for shard_path, row_idx in pointers:
            by_shard.setdefault(shard_path, []).append(row_idx)

        crops: List[np.ndarray] = []
        metas: list = []
        for shard_path, row_idxs in by_shard.items():
            table = pq.read_table(shard_path, columns=cols_needed)
            for ri in row_idxs:
                row = {c: table.column(c)[ri].as_py() for c in cols_needed}

                if config == "copgen":
                    seed_num = int(row["sample_id"].replace("seed_", ""))
                    if seed_filter is not None and seed_num not in seed_filter:
                        continue
                    meta = seed_num
                elif config == "terramind":
                    meta = int(row["sample_id"].replace("seed_", ""))
                else:  # real
                    meta = row.get("date", "")

                arr = _row_to_tensor(row, self.grid)

                if filter_valid:
                    b02 = arr[BANDS.index("B02")]
                    valid = float(np.count_nonzero(b02)) / b02.size
                    if valid < MIN_CENTRE_VALID:
                        continue

                crops.append(arr)
                metas.append(meta)

        if not crops:
            return np.zeros((0, 12, MODEL_PX, MODEL_PX), dtype=np.float32), []

        stack = np.stack(crops)
        if sort_by_seed:
            order = np.argsort(metas)
            stack = stack[order]
            metas = [metas[i] for i in order]
        return stack, metas

    def iter_cells(self) -> Iterator[CellSample]:
        """Iterate over all cells that survive the validity filter."""
        for cell_id in self.list_cells():
            sample = self.load_cell(cell_id)
            if sample is not None:
                yield sample
