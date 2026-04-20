"""
Deterministic, PYTHONHASHSEED-independent seed selection for the
COP-GEN stochastic benchmark.

COP-GEN was generated with 33 random seeds per grid cell, but the
benchmark compares against 16 real S2 L2A acquisitions and 16 TerraMind
samples, so for a fair like-for-like comparison we use 16 COP-GEN seeds
per cell.

The canonical selection is stored in ``copgen_seed_selection.json``
(shipped alongside this module). It was generated once using the SHA-256
rule defined below and should not change thereafter; this module merely
loads and validates the stored selection.

If you need to regenerate the selection (e.g.\ the pool of available seeds
changes), call :func:`generate_selection` and commit the resulting JSON.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List

#: Version tag mixed into the hash. Bumping this would re-randomise the
#: selection for the same cell IDs, so only change it if the benchmark
#: definition materially changes.
HASH_VERSION = "copgen-benchmark/v1"

#: Number of COP-GEN seeds to retain per cell.
N_SEEDS = 16

_HERE = Path(__file__).resolve().parent
SELECTION_PATH = _HERE / "copgen_seed_selection.json"


# ---------------------------------------------------------------------------
# Selection rule
# ---------------------------------------------------------------------------

def _cell_rng(cell_id: str) -> random.Random:
    """Stable per-cell RNG seeded by SHA-256 of ``HASH_VERSION:cell_id``.

    Independent of PYTHONHASHSEED, Python version, and process identity.
    """
    payload = f"{HASH_VERSION}:{cell_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    seed = int.from_bytes(digest[:8], "big")
    return random.Random(seed)


def select_seeds(cell_id: str, pool: List[int], n: int = N_SEEDS) -> List[int]:
    """Return a sorted subset of ``n`` seeds from ``pool`` for ``cell_id``.

    The choice is reproducible given ``cell_id`` and the seed pool.
    """
    if len(pool) <= n:
        return sorted(pool)
    rng = _cell_rng(cell_id)
    return sorted(rng.sample(sorted(pool), n))


# ---------------------------------------------------------------------------
# JSON storage
# ---------------------------------------------------------------------------

def generate_selection(
    cell_ids: List[str],
    pool: List[int],
    n: int = N_SEEDS,
) -> Dict[str, List[int]]:
    """Generate the canonical selection mapping ``cell_id -> [seed, ...]``.

    Call once, then serialise the result with :func:`save_selection` and
    commit it to the repository as the frozen canonical selection.
    """
    return {cell: select_seeds(cell, pool, n) for cell in sorted(cell_ids)}


def save_selection(selection: Dict[str, List[int]], path: Path = SELECTION_PATH) -> None:
    path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")


def load_selection(path: Path = SELECTION_PATH) -> Dict[str, List[int]]:
    """Load the canonical seed selection from JSON.

    This is what the benchmark should use at runtime. The JSON is the
    single source of truth — modifying the RNG logic does not change the
    canonical selection unless :func:`save_selection` is explicitly
    re-run and the new JSON is committed.
    """
    with open(path) as f:
        return json.load(f)
