# Paper Figures

Scripts and datasets for generating publication figures comparing COP-GEN and TerraMind.

---

## 📁 Folder Structure

```
paper_figures/
├── scripts/                          # Batch experiment scripts
├── spectral_profiles/                # Output: spectral profile figures per tile
├── lat-lon-thumbnails/               # Output: location thumbnails
├── paper_figures_datasets/           # Input datasets and experiment outputs
│   ├── all_tiles_dataset/            # Multi-tile benchmark dataset
│   ├── one_tile_clean/               # Single-tile clean datasets
│   ├── one_tile_distribution_narrowing/  # Narrowing distribution experiments
│   ├── one_tile_datasets_DEM_LULC_to_S2L2A/  # DEM+LULC→S2L2A experiments
│   ├── climate_zones_copgen/         # Climate zone experiments (COP-GEN)
│   ├── climate_zones_terramind/      # Climate zone experiments (TerraMind)
│   └── architecture_figure/          # Architecture diagram samples
├── population_cache/                 # Cached population density data
├── treecover_cache/                  # Cached tree cover data
└── *.py                              # Figure generation scripts
```

---

## 🎨 Figure Generation Scripts

### 1. Spectral Profiles (`spectral_profiles.py`)

Compares S2L2A spectral signatures (per LULC class) between GT, COP-GEN, and TerraMind.

```bash
python paper_figures/spectral_profiles.py \
    --tile-root ./paper_figures/paper_figures_datasets/one_tile_datasets_DEM_LULC_to_S2L2A/143D_1481R \
    --terramind-exp input_DEM_LULC_output_S2L2A_seed_111 \
    --copgen-exp input_DEM_LULC_cloud_mask_output_S1RTC_S2L1C_S2L2A_lat_lon_timestamps_seed_111 \
    --seed 42
```

**Outputs:** `spectral_profiles/<tile>_seed<N>/best-{copgen,terramind,per-model}/`
- `spectral_profiles.png` — All-class spectral comparison
- `spectral_profiles_by_class.png` — Per-class breakdown
- `{GT,Copgen,Terramind}_S2L2A_{raw,annotated}.png` — RGB visualizations

---

### 2. Climate Zones (`climate_zones_plotting.py`)

Plots generated lat/lon predictions on world maps with Köppen climate zones, population density, or tree cover backgrounds.

```bash
python3 paper_figures/climate_zones_plotting.py \
    --comparison \                                            
    --copgen-root ./paper_figures/paper_figures_datasets/climate_zones_copgen \
    --terramind-root ./paper_figures/paper_figures_datasets/climate_zones_terramind \
    --save ./paper_figures/paper_figures_datasets \
    --population-raster-path ./paper_figures/ppp_2020_1km_Aggregated.tif \
    --population-res-deg 0.1 \
    --scatter-size 45 \
    --basemap population \
    --classes built_area
```

**Outputs:** `paper_figures_datasets/climate_zones_*/lat_lon_by_lulc_<class_name>.png`

---

### 3. Narrowing Distributions (`narrowing_distributions.py`)

Visualizes how S2L2A output distributions narrow as more conditioning modalities are added.

```bash
python3 paper_figures/narrowing_distributions.py ./paper_figures/paper_figures_datasets/one_tile_distribution_narrowing/95U_112R
python3 paper_figures/narrowing_distributions.py ./paper_figures/paper_figures_datasets/one_tile_distribution_narrowing/195D_669L
python3 paper_figures/narrowing_distributions.py ./paper_figures/paper_figures_datasets/one_tile_distribution_narrowing/211D_500R
python3 paper_figures/narrowing_distributions.py ./paper_figures/paper_figures_datasets/one_tile_distribution_narrowing/215U_1019L
python3 paper_figures/narrowing_distributions.py ./paper_figures/paper_figures_datasets/one_tile_distribution_narrowing/248U_978R
python3 paper_figures/narrowing_distributions.py ./paper_figures/paper_figures_datasets/one_tile_distribution_narrowing/250U_409R
python3 paper_figures/narrowing_distributions.py ./paper_figures/paper_figures_datasets/one_tile_distribution_narrowing/256U_1125L
python3 paper_figures/narrowing_distributions.py ./paper_figures/paper_figures_datasets/one_tile_distribution_narrowing/272D_1525R
```

**Outputs:** `<tile>/narrowing_histograms/`
- `*_stacked.png` — Stacked distribution plots
- Per-scenario histograms

*Helper script for cropping stacked histograms:*
(Post-processes stacked histogram images by cropping margins)
```bash
python3 paper_figures/crop_stacked_narrow_distribution.py ./paper_figures/paper_figures_datasets/one_tile_distribution_narrowing/195D_669L/narrowing_histograms
```

---

### 4. Lat/Lon Comparison (`lat_lon_comparison.py`)

Compares GT vs COP-GEN vs TerraMind lat/lon predictions on Köppen climate zone maps.

NOTE: The difference between `lat_lon_comparison.py` and `climate_zones_plotting.py` is that `lat_lon_comparison.py` is used to compare the lat/lon predictions of a specific DEM+LULC tile, while `climate_zones_plotting.py` is used to compare the lat/lon predictions based solely on a LULC input class (e.g. the tile is all "water").

```bash
python3 paper_figures/lat_lon_comparison.py ./paper_figures/paper_figures_datasets/one_tile_datasets_DEM_LULC_timestamps_to_lat_lon/215U_1019L --highlight 380U_1238R --vis-gridcells
```

**Outputs:** `<tile>/outputs/lat_lon_comparison.png`

---

### 5. Lat/Lon Thumbnails (`lat-lon-thumbnail.py`)

Creates 512×512 thumbnails visualisations of a location given a grid cell.

```bash
python3 ./paper_figures/lat-lon-thumbnail.py 502U_263R --koppen-path ./paper_figures/koppen_2020.geojson --zoom-deg 29 --output ./paper_figures/lat-lon-thumbnails

```

**Outputs:** `lat-lon-thumbnails/lat_lon_thumbnail_*.png`

---

### 6. Custom Dataset Creation (`custom_dataset.py`)

Extracts specific tiles from the full test dataset for focused experiments.

**Single tile:**
```bash
python paper_figures/custom_dataset.py \
    --root_dataset ./data/majorTOM/test_dataset_copgen \
    --output_dataset ./paper_figures/paper_figures_datasets/one_tile_clean/143D_1481R \
    --grid_ids 143D_1481R
```

**Batch create multiple one-tile datasets:**
```bash
TILES=("143D_1481R" "95U_112R" "195D_669L" "211D_500R" "215U_1019L" "248U_978R" "250U_409R" "256U_1125L" "272D_1525R" "227D_564L")

ROOT="./data/majorTOM/test_dataset_copgen"
OUT="./paper_figures/paper_figures_datasets/one_tile_datasets"

for TILE in "${TILES[@]}"; do
    echo "Processing $TILE ..."
    python3 paper_figures/custom_dataset.py \
        --root_dataset "$ROOT" \
        --output_dataset "$OUT/$TILE" \
        --grid_ids "$TILE"
done
```

### 7. Band infilling experiments

We can use the `copgen_cli_bands.py` script to run experiments with specific band infilling (note that in `copgen_cli.py` we can only specify the base modalities, not the band-level modalities).

```bash
python copgen_cli_bands.py \
    --model ./models/copgen/cop_gen_base/500000_nnet_ema.pth \
    --config configs/copgen/discrete/cop_gen_base.py \
    --dataset-root ./paper_figures/paper_figures_datasets/architecture_figure/560U_34R \
    --input-keys S2L1C_B02_B03_B04_B08 \
    --batch-size 1 \
    --vis-every 1
```

---

## 🔧 Batch Scripts (`scripts/`)

| Script | Description |
|--------|-------------|
| `spectral_profiles.sh` | Generate spectral profiles for all paper tiles |
| `narrowing_distribution_copgen.sh` | COP-GEN narrowing distribution experiments |
| `narrowing_distribution_terramind.sh` | TerraMind narrowing distribution experiments |
| `climate_zones.sh` | Generate climate zone lat/lon predictions |
| `lat_lon_complementary.sh` | Lat/lon conditioning with alternate coordinates |
| `seasonality_complementary.sh` | Timestamp swap experiments for seasonality |
| `make_summary.sh` | Extract MAE values and copy PNGs for comparison |

### Example: Run All Spectral Profiles

```bash
bash paper_figures/scripts/spectral_profiles.sh
```

### Example: Run Narrowing Distribution (COP-GEN)

```bash
bash paper_figures/scripts/narrowing_distribution_copgen.sh
```

---

## 📊 Key Datasets

| Dataset | Description |
|---------|-------------|
| `one_tile_clean/` | Single tiles with clean GT data |
| `one_tile_distribution_narrowing/` | Experiments with progressive conditioning |
| `one_tile_datasets_DEM_LULC_to_S2L2A/` | DEM+LULC → S2L2A generation |
| `one_tile_datasets_DEM_LULC_timestamps_to_lat_lon/` | Location prediction experiments |
| `climate_zones_copgen/` | Per-LULC-class lat/lon generation (COP-GEN) |
| `climate_zones_terramind/` | Per-LULC-class lat/lon generation (TerraMind) |
| `one_tile_dataset_latlon_seasonality/` | Seasonality experiments with lat/lon |
| `one_tile_dataset_latlon_timestamp_seasonality/` | Seasonality with lat/lon + timestamps |

---

## 🌍 External Data Files

| File | Description |
|------|-------------|
| `koppen_2020.geojson` | Köppen-Geiger climate classification |
| `ppp_2020_1km_Aggregated.tif` | WorldPop population density (~829MB) |
| `world.topo.bathy.*.jpg` | NASA topo/bathymetry basemap |
| `ne_*.zip` | Natural Earth shapefiles (countries, lakes, oceans) |
| `treecover_cache/` | Global tree cover tiles |

---

## 📝 Quick Reference

### Generate Multiple Samples (COP-GEN)

**Single tile:**
```bash
SEEDS=(111 222 333 444 555 666 777 888)
for SEED in ${SEEDS[@]}; do
    python3 copgen_cli.py \
        --dataset-root ./paper_figures/paper_figures_datasets/one_tile_clean/143D_1481R \
        --model ./models/copgen/cop_gen_base/500000_nnet_ema.pth \
        --config ./configs/copgen/discrete/cop_gen_base.py \
        --seed $SEED \
        --inputs DEM LULC cloud_mask \
        --batch-size 1 --vis-every 1
done
```

**All tiles:**
```bash
TILES=("143D_1481R" "95U_112R" "195D_669L" "211D_500R" "215U_1019L" "248U_978R" "250U_409R" "256U_1125L" "272D_1525R")
SEEDS=(111 222 333 444 555 666 777 888)
MODEL=./models/copgen/cop_gen_base/500000_nnet_ema.pth
CONFIG=./configs/copgen/discrete/cop_gen_base.py
ROOT=./paper_figures/paper_figures_datasets/one_tile_datasets

for TILE in "${TILES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        python3 copgen_cli.py \
            --dataset-root "$ROOT/$TILE/" \
            --model "$MODEL" --config "$CONFIG" \
            --seed "$SEED" \
            --inputs DEM LULC cloud_mask \
            --batch-size 1 --vis-every 1
    done
done
```

### Generate Multiple Samples (TerraMind)

**Single tile:**
```bash
SEEDS=(111 222 333 444 555 666 777 888)
for SEED in ${SEEDS[@]}; do
    python3 terramind_cli.py \
        --dataset_root ./paper_figures/paper_figures_datasets/one_tile_clean/143D_1481R \
        --input DEM LULC \
        --output S2L2A \
        --seed $SEED --visualize
done
```

**All tiles:**
```bash
TILES=("143D_1481R" "95U_112R" "195D_669L" "211D_500R" "215U_1019L" "248U_978R" "250U_409R" "256U_1125L" "272D_1525R")
SEEDS=(111 222 333 444 555 666 777 888)
ROOT=./paper_figures/paper_figures_datasets/one_tile_datasets

for TILE in "${TILES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        python3 terramind_cli.py \
            --dataset_root "$ROOT/$TILE" \
            --input DEM LULC \
            --output S2L2A \
            --seed "$SEED" --visualize
    done
done
```

### Summarize Experiment MAE Values

```bash
bash paper_figures/scripts/make_summary.sh paper_figures/paper_figures_datasets/one_tile_datasets_DEM_LULC_to_S2L2A
```

---

## 📚 Additional Documentation

- `seasonality_readme.md` — Detailed seasonality experiment notes with tile metadata

