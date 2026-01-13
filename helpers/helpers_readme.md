# Helpers

Utility scripts for test set visualization and performance benchmarking.

## Folder Structure

```
helpers/
├── test_set/
│   ├── visualise_set.py         # Plot grid-cell coverage on world map
│   ├── plot_test_grid_cells.py  # Plot test cells with labels
│   ├── txts/                    # Grid cell lists
│   │   ├── train.txt
│   │   ├── test.txt
│   │   └── hold_out.txt
│   └── imgs/                    # Generated visualizations
└── performance/
    └── eval_copgen_quant_timings.py  # COP-GEN evaluation benchmarks
```

## Test Set Visualization

### Visualize dataset coverage

Plot train/test/holdout splits on a world map:

```bash
cd helpers/test_set
python visualise_set.py \
  --inputs ./txts/train.txt ./txts/hold_out.txt ./txts/test.txt \
  --output ./imgs/coverage.png \
  --colors "#2E86AB" "#E69F00" "#E83417" \
  --style light
```

Add `--output_html ./imgs/coverage.html` for an interactive Folium map.

### Plot test grid cells

```bash
cd helpers/test_set
python plot_test_grid_cells.py ./txts/test.txt --save ./imgs/plot_test.png
```

## Performance Benchmarking

Benchmark COP-GEN evaluation pipeline with detailed timing breakdown:

```bash
python helpers/performance/eval_copgen_quant_timings.py \
  --config ./configs/copgen/copgen_eval/config.py \
  --nnet_path ./models/copgen/nnet_ema.pth \
  --data_path ./data/latents \
  --target LULC_LULC \
  --generate LULC_LULC,DEM_DEM \
  --condition S2L2A_B02_B03_B04_B08 \
  --batch_size 8
```

---

## Test Set Creation (Internal)

<details>
<summary>Click to expand internal test set creation notes</summary>

### Select random subset

```bash
python3 -c 'import random, sys; random.seed(1234); print("".join(random.sample(open("./txts/hold_out.txt").readlines(), 1000)), end="")' > ./txts/test.txt
```

### Extract TIFFs from tar.gz archives

See the Python script in the original documentation for extracting tiles from MajorTOM archives using parallel processing.

### Encode dataset into latents

Use `encode_moments_vae.py` with appropriate config files for each modality (DEM, LULC, S2L2A, S2L1C, S1RTC, cloud_mask).

### Merge LMDB latents

```bash
python scripts/merge_lmdbs.py \
    --input_dir ${LATENTS_PATH}/test \
    --output_dir ${MERGED_LATENTS_PATH}/test \
    --batch_size 20000
```

</details>
