# Benchmark

Benchmarking code for comparing COP-GEN and TerraMind geospatial foundation models.

## Folder Structure

```
benchmark/
├── scripts/                    # Bash scripts for running experiments
│   ├── full-copgen.sh          # Full COP-GEN experiments
│   ├── full-terramind.sh       # Full TerraMind experiments
│   ├── cop-gen-leave-one-out.sh
│   └── terramind-leave-one-out.sh
├── tables/                     # Generate terminal summaries for LaTeX
│   ├── leave_one_out.py        # Leave-one-out ablation tables
│   └── best-of-n.py            # Best-of-N cross-modal tables
├── evaluation/                 # Metrics and evaluation
│   ├── evaluate_copgen_run.py
│   ├── evaluate_terramind_run.py
│   ├── metrics.py
│   └── scripts/                # Per-tile evaluation scripts
├── generations/                # Model inference utilities
│   ├── copgen.py
│   └── terramind.py
├── common/                     # Shared constants (modalities, paths)
├── dataloaders/                # Dataset classes
├── io/                         # TIFF and coords I/O helpers
├── utils/                      # Bandstacker, logging, plotting
└── visualize/                  # Visualization tools
```

## Running Experiments

### TerraMind

```bash
python terramind_cli.py --dataset_root <path> --input S2L2A S1RTC --output LULC
```

- **Inputs:** DEM, LULC, S1RTC, S2L2A, S2L1C, coords (one or more)
- **Output:** exactly one modality

Real example:
```bash
python terramind_cli.py --input DEM LULC S1RTC coords \
                        --output S2L2A \
                        --dataset_root ./paper_figures/paper_figures_datasets/architecture_figure/502U_263R \
                        --seed 1234 \
                        --visualize
```

### COP-GEN

```bash
python copgen_cli.py \
  --dataset-root <path> \
  --model <path/to/nnet_ema.pth> \
  --config <path/to/config.py> \
  --inputs lat_lon DEM LULC S1RTC S2L2A
```

- **Inputs:** lat_lon, timestamps, DEM, LULC, S1RTC, S2L1C, S2L2A, cloud_mask
- **Outputs:** all modalities not in inputs

Real example:
```bash
python3 copgen_cli.py \
            --dataset-root paper_figures/paper_figures_datasets/architecture_figure/560U_34R \
            --model ./models/copgen/cop_gen_base/500000_nnet_ema.pth \
            --config ./configs/copgen/discrete/cop_gen_base.py \                        
            --seed 111 \                
            --inputs cloud_mask LULC lat_lon \
            --batch-size 1 \
            --vis-every 1
```

### Batch Experiments

```bash
# Full (all modalities minus predicted) experiments
bash benchmark/scripts/full-terramind.sh
bash benchmark/scripts/full-copgen.sh
# Leave-one-out experiments (paper tables)
bash benchmark/scripts/terramind-leave-one-out.sh
bash benchmark/scripts/cop-gen-leave-one-out.sh
```

## Generating Summary Tables

Print terminal summaries of experiment results (for LaTeX):

```bash
# Leave-one-out ablation results
python benchmark/tables/leave_one_out.py --dataset_root <path>
# python3 benchmark/tables/leave_one_out.py --dataset_root ./data/majorTOM/test_dataset_copgen_leave_one_out

# Best-of-N cross-modal results  
python benchmark/tables/best-of-n.py --dataset_root <path>
# python3 benchmark/tables/best-of-n.py --dataset_root ./data/majorTOM/test_dataset_copgen
```

Optional: `--export_csv <path>` to save as CSV.

## Evaluation

Evaluate a single run:

```bash
python benchmark/evaluation/evaluate_terramind_run.py \
  --dataset-dir <path> --run-dir <path/to/run> --per-tile

python benchmark/evaluation/evaluate_copgen_run.py \
  --dataset-dir <path> --run-dir <path/to/run> --per-tile
```

Example:
```bash
# Copgen
DATASET_ROOT="./data/majorTOM/test_dataset_copgen_leave_one_out"
COPGEN_DIR="$DATASET_ROOT/outputs/copgen"                             
        
for folder in "$COPGEN_DIR"/*; do                            
    if [ -d "$folder" ]; then            
        echo "Running evaluation for: $folder"
        python3 benchmark/evaluation/evaluate_copgen_run.py \   
            --dataset-root $DATASET_ROOT \
            --run-root "$folder" \
            --per-tile
    fi
done


# Terramind
python -m benchmark.evaluation.evaluate_terramind_run \
        --dataset-dir ./data/majorTOM/test_dataset_copgen_leave_one_out \
        --run-dir ./data/majorTOM/test_dataset_copgen_leave_one_out/outputs/terramind/input_coords_LULC_S1RTC_S2L1C_output_DEM_seed_23 \
        --verbose --per-tile

```

Results are saved to `output_metrics.txt` and `output_metrics_per_tile.csv` in the run directory.
