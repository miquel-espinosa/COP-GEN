from __future__ import annotations

import argparse
from pathlib import Path
from benchmark.generations.copgen import CopgenBatchGenerator
from benchmark.evaluation.evaluate_copgen_run import evaluate_copgen_run


def parse_args():
    p = argparse.ArgumentParser(
        description="Run COP-GEN batch generation (native-res, MajorTOM layout)",
        epilog="Available modalities: lat_lon, timestamps, DEM, LULC, S1RTC, S2L1C, S2L2A, cloud_mask"
    )
    p.add_argument("--dataset-root", required=False, type=str, help="MajorTOM dataset root (defaults to <root>/data/cop-gen-small-test)")
    p.add_argument("--model", required=True, type=str, help="Path to Copgen model checkpoint (nnet_ema.pth)")
    p.add_argument("--config", required=True, type=str, help="Path to Copgen model config .py (get_config())")
    p.add_argument("--inputs", required=True, nargs="+", type=str, help="Condition base modalities (e.g. S2L2A S1RTC DEM LULC). Available: lat_lon, timestamps, DEM, LULC, S1RTC, S2L1C, S2L2A, cloud_mask")
    p.add_argument(
        "--inputs-paths",
        required=False,
        nargs="+",
        type=str,
        help=(
            "Optional overrides for the listed --inputs. "
            "Pass either positional values (same order as --inputs) or explicit mappings like 'DEM=/path/file.tif' or 'lat_lon=-32.4,152.0'. "
            "For lat_lon use 'lat,lon'. For timestamps use 'dd-mm-yyyy'. "
            "For LULC/cloud_mask you can pass a class name like 'water' or a .tif path. "
            "Other modalities expect a .tif path."
        ),
    )
    p.add_argument("--seed", required=False, type=int, default=1234, help="Random seed for reproducibility")
    p.add_argument("--max", dest="max_products", required=False, type=int, default=None, help="Limit number of products")
    p.add_argument("--batch-size", type=int, default=4, help="Batch size for model.generate")
    p.add_argument("--vis-every", type=int, default=0, help="Visualise every N batches (0 to disable)")
    p.add_argument("--visualise-histograms", action="store_true", help="Visualise histograms")
    p.add_argument("--samples", type=int, default=1, help="Number of samples to generate per condition")
    return p.parse_args()


def main():
    args = parse_args()
    
    # Define allowed modalities
    allowed_modalities = {"lat_lon", "timestamps", "DEM", "LULC", "S1RTC", "S2L1C", "S2L2A", "cloud_mask"}
    
    # Validate inputs
    invalid_inputs = [m for m in args.inputs if m not in allowed_modalities]
    if invalid_inputs:
        raise ValueError(f"Invalid input modalities: {invalid_inputs}. Allowed: {sorted(allowed_modalities)}")
    
    def _norm(name: str) -> str:
        return name.strip().lower()

    input_overrides = None
    if args.inputs_paths is not None:
        explicit_overrides = {}
        positional_values = []

        for raw in args.inputs_paths:
            if "=" in raw:
                base_candidate, value = raw.split("=", 1)
                base_candidate = base_candidate.strip()
                value = value.strip()
                match = [b for b in args.inputs if _norm(b) == _norm(base_candidate)]
                if not match:
                    raise ValueError(f"Unknown modality '{base_candidate}' in --inputs-paths (expected one of {args.inputs}).")
                base_key = match[0]
                if base_key in explicit_overrides:
                    raise ValueError(f"Duplicate override provided for modality '{base_key}'.")
                explicit_overrides[base_key] = value
            else:
                positional_values.append(raw)

        remaining_bases = [b for b in args.inputs if b not in explicit_overrides]
        if len(positional_values) > len(remaining_bases):
            raise ValueError(
                f"Too many positional entries for --inputs-paths. "
                f"Got {len(positional_values)} values for {len(remaining_bases)} remaining modalities."
            )
        for base, value in zip(remaining_bases, positional_values):
            explicit_overrides[base] = value

        input_overrides = explicit_overrides if explicit_overrides else None
    
    # Compute outputs as all modalities minus inputs
    outputs = list(allowed_modalities - set(args.inputs))
    
    # Print summary
    print("=" * 60)
    print("COP-GEN GENERATION CONFIGURATION")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Config: {args.config}")
    print(f"Dataset root: {args.dataset_root}")
    print(f"Seed: {args.seed}")
    print(f"Batch size: {args.batch_size}")
    print(f"Samples per condition: {args.samples}")
    print(f"Max products: {args.max_products if args.max_products else 'All'}")
    print(f"Visualisation: Every {args.vis_every} batches" if args.vis_every > 0 else "Visualisation: Disabled")
    print(f"\nInput modalities (conditions): {', '.join(sorted(args.inputs))}")
    print(f"Output modalities (to generate): {', '.join(sorted(outputs))}")
    if input_overrides:
        print("Input overrides provided (modality -> value/path):")
        for k, v in input_overrides.items():
            print(f"  - {k}: {v}")
    print("=" * 60)
    print()
    
    
    gen = CopgenBatchGenerator(
        model_path=Path(args.model),
        config_path=Path(args.config),
        dataset_root=Path(args.dataset_root),
        seed=args.seed,
        input_overrides=input_overrides,
    )
    gen.process_all(
        condition_bases=args.inputs,
        generate_bases=outputs,
        max_products=args.max_products,
        batch_size=args.batch_size,
        vis_every=(args.vis_every if args.vis_every > 0 else None),
        vis_comparison=True,
        visualise_histograms=args.visualise_histograms,
        samples_per_condition=args.samples,
    )

    # Auto-run evaluation after generation
    run_name = f"input_{'_'.join(sorted(args.inputs))}_output_{'_'.join(sorted(outputs))}_seed_{args.seed}"
    dataset_root = Path(args.dataset_root)
    run_root = dataset_root / "outputs" / "copgen" / run_name
    metrics = evaluate_copgen_run(dataset_root, run_root, verbose=True)
    print("\nSaved evaluation report to:", run_root / "output_metrics.txt")


if __name__ == "__main__":
    main()


