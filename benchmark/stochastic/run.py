"""
Reproducible entry point for the COP-GEN stochastic benchmark.

Downloads the three ``Major-TOM/COP-GEN-Benchmark`` configs from
HuggingFace, extracts the documented 192x192 evaluation footprint from
each 1056x1056 tile, runs all perceptual (Stream 1) and physical
(Stream 2) metrics, and writes a per-cell CSV plus a mean-\u00b1-std summary.

Usage:
    python -m benchmark.stochastic.run --output metrics.csv

Options:
    --hf-dataset   HuggingFace repo ID (default: Major-TOM/COP-GEN-Benchmark)
    --cache-dir    local HF cache dir   (default: ~/.cache/huggingface)
    --limit-cells  restrict to first N cells for debugging
    --device       "cuda" or "cpu" for embedding nets (default: cuda if available)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .loader import BenchmarkLoader, CellSample, BANDS
from .metrics.perceptual import (
    intra_set_distance, nn_accuracy, precision_recall,
)
from .metrics.physical import (
    mmd, spectral_pool, spectral_range_coverage, wasserstein_per_band,
)
from .metrics.embeddings import LPIPSDistance, ResNet50Embedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-source metrics
# ---------------------------------------------------------------------------

def metrics_for_pair(
    real: np.ndarray,
    fake: np.ndarray,
    prefix: str,
    resnet: ResNet50Embedder,
    lpips_model: LPIPSDistance,
) -> Dict:
    """Compute all metrics for a (real, generated) pair across all three
    embedding representations + physical stream."""
    out: Dict = {
        f"{prefix}_n_real": len(real),
        f"{prefix}_n_fake": len(fake),
    }
    if len(real) < 2 or len(fake) < 2:
        return out

    # -- Spectral embedding (Stream 1 + Stream 2 input) --
    real_spec = spectral_pool(real)
    fake_spec = spectral_pool(fake)

    out[f"{prefix}_spec_nn_accuracy"] = nn_accuracy(real_spec, fake_spec)
    p, r = precision_recall(real_spec, fake_spec, k=5)
    out[f"{prefix}_spec_precision"] = p
    out[f"{prefix}_spec_recall"] = r
    out[f"{prefix}_spec_intra_set_dist"] = intra_set_distance(fake_spec)
    out["real_spec_intra_set_dist"] = intra_set_distance(real_spec)

    # -- ResNet-50 RGB embedding --
    real_emb = resnet.embed(real)
    fake_emb = resnet.embed(fake)
    out[f"{prefix}_rgb_nn_accuracy"] = nn_accuracy(real_emb, fake_emb)
    p_r, r_r = precision_recall(real_emb, fake_emb, k=5)
    out[f"{prefix}_rgb_precision"] = p_r
    out[f"{prefix}_rgb_recall"] = r_r
    out[f"{prefix}_rgb_intra_set_dist"] = intra_set_distance(fake_emb)
    out["real_rgb_intra_set_dist"] = intra_set_distance(real_emb)

    # -- LPIPS intra-set distance --
    out[f"{prefix}_lpips_intra_set"] = lpips_model.intra_set_distance(fake)
    out["real_lpips_intra_set"] = lpips_model.intra_set_distance(real)

    # -- Stream 2: physical --
    out[f"{prefix}_mmd"] = mmd(real_spec, fake_spec)
    w_per_band = wasserstein_per_band(real_spec, fake_spec)
    out[f"{prefix}_wasserstein_mean"] = float(w_per_band.mean())
    out[f"{prefix}_wasserstein_b04"] = float(w_per_band[3])
    out[f"{prefix}_wasserstein_b08"] = float(w_per_band[7])
    cov = spectral_range_coverage(real_spec, fake_spec)
    out[f"{prefix}_coverage_mean"] = float(np.nanmean(cov))

    return out


def process_cell(sample: CellSample, resnet, lpips_model) -> Dict:
    row: Dict = {"cell_id": sample.cell_id, "n_real": len(sample.real)}
    row.update(metrics_for_pair(sample.real, sample.copgen, "copgen", resnet, lpips_model))
    row.update(metrics_for_pair(sample.real, sample.terramind, "terramind", resnet, lpips_model))
    return row


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    print()
    print("=" * 70)
    print(f"SUMMARY (mean \u00b1 std across {len(df)} cells)")
    print("=" * 70)
    exclude = {"cell_id", "n_real",
               "copgen_n_real", "copgen_n_fake",
               "terramind_n_real", "terramind_n_fake"}
    for col in sorted(c for c in df.columns if c not in exclude):
        vals = df[col].dropna()
        if len(vals) and pd.api.types.is_numeric_dtype(vals):
            print(f"  {col:<40s} {vals.mean():>10.4f}  \u00b1  {vals.std():>10.4f}  (n={len(vals)})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hf-dataset", default="Major-TOM/COP-GEN-Benchmark",
                        help="HuggingFace dataset ID")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="Local HF cache dir (default: ~/.cache/huggingface)")
    parser.add_argument("--local-root", type=Path, default=None,
                        help="Read parquets from a local mirror of the HF "
                             "dataset layout (real/data/*.parquet, "
                             "copgen/data/*.parquet, terramind/data/*.parquet, "
                             "metadata/benchmark_grid.json) instead of "
                             "downloading from HuggingFace.")
    parser.add_argument("--output", type=Path, default=Path("metrics_per_cell.csv"))
    parser.add_argument("--limit-cells", type=int, default=None,
                        help="Only process the first N cells (debugging)")
    parser.add_argument("--device", default=None, help="\"cuda\" or \"cpu\"")
    args = parser.parse_args()

    if args.device is None:
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"

    loader = BenchmarkLoader(
        hf_dataset=args.hf_dataset,
        cache_dir=args.cache_dir,
        max_cells=args.limit_cells,   # stops shard downloads early for smoke tests
        local_root=args.local_root,
    )

    cells = loader.list_cells()
    if args.limit_cells:
        cells = cells[:args.limit_cells]
    log.info("Processing %d cells on %s", len(cells), args.device)

    log.info("Loading ResNet-50 (ImageNet) and LPIPS (AlexNet) ...")
    resnet = ResNet50Embedder(device=args.device)
    lpips_model = LPIPSDistance(device=args.device)

    rows = []
    for cell_id in tqdm(cells, desc="cells"):
        sample = loader.load_cell(cell_id)
        if sample is None:
            continue
        rows.append(process_cell(sample, resnet, lpips_model))

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    log.info("Saved %d rows to %s", len(df), args.output)

    print_summary(df)


if __name__ == "__main__":
    main()
