# Debugging scripts

## STD evaluation

Run a STD evaluation for a given set of inputs and outputs.

```bash
python legacy_evaluations/copgen_std.py \
  --dataset-root ./data/majorTOM/cop-gen-small-test \
  --model ./models/copgen/cop_gen_huge/185000_nnet_ema.pth \
  --config ./configs/copgen/discrete/cop_gen_huge.py \
  --inputs S1RTC cloud_mask lat_lon timestamps \
  --seed 1234 \
  --num-tiles 2 \
  --num-samples 1000 \
  --vis-every 100
```

## Rank samples

Rank the samples for a given tile based on the evaluation metrics.

```bash
python legacy_evaluations/rank_samples.py \
  --tile-root data/majorTOM/cop-gen-small-test/outputs/copgen_std/input_S1RTC_cloud_mask_lat_lon_timestamps_output_DEM_LULC_S2L1C_S2L2A_seed_1234_tiles_2_samples_100/621U_27L \
  --topk 5
```

## Plot sample

Given lots of generations for a given tile, we can plot a specific sample (e.g. one of the top-k samples)

```bash
python legacy_evaluations/plot_sample.py \
  --sample-root /path/to/outputs/copgen_std/.../tile_id/samples/sample_14 \
  --config configs/copgen/discrete/cop_gen_base.py \
  --milestone 0
```