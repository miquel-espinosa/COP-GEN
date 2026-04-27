# COP-GEN Stochastic Benchmark — Reproducible Evaluation

This directory contains a self-contained, reproducible implementation of the
stochastic benchmark reported in the COP-GEN paper. It downloads the public
dataset from HuggingFace and runs all metrics end-to-end.

Third parties can clone the repository, install the dependencies, and run
the benchmark with a single command to reproduce the paper's Table 1.

## Layout

```
benchmark/
├── README.md                    This file.
├── requirements.txt             Pinned Python dependencies.
├── copgen_seed_selection.json   Canonical 16-out-of-33 COP-GEN seed
                                 selection per cell (SHA-256 derived).
├── footprint.py                 Centre-crop utility for the 192x192
                                 evaluation window.
├── seeds.py                     Deterministic seed selection helper.
├── loader.py                    HuggingFace streaming loader.
├── run.py                       Main entry point.
└── metrics/
    ├── perceptual.py            1-NN, Precision/Recall, intra-set distance.
    ├── physical.py              MMD, Wasserstein, spectral-range coverage.
    └── embeddings.py            ResNet-50 (ImageNet) + LPIPS (AlexNet).
```

## Quick start

```bash
pip install -r benchmark/requirements.txt
python -m benchmark.stochastic.run --output metrics.csv
```

Expected runtime: ~20 minutes on a single A100 GPU (dominated by the RGB
ResNet-50 and LPIPS forward passes). First run downloads ~72 GB from
HuggingFace and caches it locally; subsequent runs are faster.

Expected output:

* `metrics.csv` — one row per benchmark cell (489 rows, 37 columns)
* A mean-\u00b1-std summary printed to stdout

Compare the printed summary to Table 1 of the paper:

* On CPU, the numbers are bit-identical across runs (deterministic).
* On GPU, ResNet-50 and LPIPS introduce small CUDA non-associativity;
  the mean-\u00b1-std values still match the paper to 3 decimal places.

## Reproducibility

* **Data.** All tensors come from the public
  [`Major-TOM/COP-GEN-Benchmark`](https://huggingface.co/datasets/Major-TOM/COP-GEN-Benchmark)
  dataset. No local paths or private files are used.
* **Footprint.** The 192x192 evaluation window is extracted from the
  1056x1056 published tiles using the exact function in `footprint.py`
  (shipped with the dataset under `metadata/benchmark_footprint.py`).
* **Seed selection.** COP-GEN was generated with 33 seeds per cell; for
  a like-for-like comparison against the 16 real and 16 TerraMind samples,
  we subsample 16 COP-GEN seeds per cell. The canonical selection is
  derived from SHA-256 hashes of the cell IDs (see `seeds.py`) and is
  frozen in `copgen_seed_selection.json`. The runtime uses the committed
  JSON rather than regenerating from the hash, so even if the hashing
  rule changes the results remain reproducible.
* **Dependencies.** Pinned minimum versions in `requirements.txt`. The
  pretrained networks (`torchvision` ResNet-50, `lpips` AlexNet) are
  fixed across versions and load deterministically.
* **RNG.** No random state is consulted at runtime. Cell ordering
  (`sorted`), seed subsample (frozen JSON), filtering threshold, and
  embedding weights are all fixed. The single `random.seed(42)` in LPIPS
  pair sampling is dormant at 16 samples per cell (120 pairs $<$ 500).

## Running on a subset

For quick sanity checks:

```bash
# 5 cells (~2 min on CPU, ~30 s on GPU after initial download)
python -m benchmark.stochastic.run --output metrics_small.csv --limit-cells 5
```

## Running against a local mirror

If you already have the dataset on disk (laid out exactly as on
HuggingFace --- `real/data/*.parquet`, `copgen/data/*.parquet`,
`terramind/data/*.parquet`, `metadata/benchmark_grid.json`), skip the
download entirely:

```bash
python -m benchmark.stochastic.run \
    --local-root /path/to/COP-GEN-Benchmark \
    --output metrics.csv
```

This is equivalent to the HuggingFace path but avoids the ~72 GB
transfer.

## Metrics

See `metrics/perceptual.py` and `metrics/physical.py` for the
implementations, and the paper for detailed definitions. All metrics are
computed per cell and aggregated as mean \u00b1 std across cells.

| Stream | Metric | File |
|--------|--------|------|
| Perceptual | 1-Nearest Neighbour accuracy | `perceptual.py` |
| Perceptual | Precision / Recall (k=5) | `perceptual.py` |
| Perceptual | Mean intra-set pairwise distance | `perceptual.py` |
| Perceptual | ResNet-50 RGB embedding | `embeddings.py` |
| Perceptual | LPIPS (AlexNet) perceptual distance | `embeddings.py` |
| Physical | Maximum Mean Discrepancy (MMD) | `physical.py` |
| Physical | 1-D Wasserstein per band | `physical.py` |
| Physical | Spectral range coverage | `physical.py` |

## Citation

If you use this benchmark, please cite the paper:

```bibtex
@article{copgen2026,
    title   = {COP-GEN: Latent Diffusion Transformer for Copernicus Earth
               Observation Data},
    author  = {Espinosa, Miguel and Gmelich Meijling, Eva and Marsocci,
               Valerio and Crowley, Elliot J. and Czerkawski, Mikolaj},
    year    = {2026},
    journal = {arXiv preprint arXiv:2603.03239},
    url     = {https://arxiv.org/abs/2603.03239},
}
```
