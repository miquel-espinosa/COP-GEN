"""
Band-level COP-GEN CLI for infilling ablations.

This script allows specifying exact modality keys (e.g., S2L1C_B02_B03_B04_B08)
instead of base modalities (e.g., S2L1C), enabling band infilling experiments.

Example usage:
    python copgen_cli_bands.py \
        --model path/to/nnet_ema.pth \
        --config configs/copgen/discrete/cop_gen_base.py \
        --dataset-root /path/to/cop-gen-small-test \
        --input-keys S2L1C_B02_B03_B04_B08 \
        --batch-size 4
"""
from __future__ import annotations

import argparse
from pathlib import Path
from benchmark.generations.copgen import CopgenBatchGenerator
from benchmark.evaluation.evaluate_copgen_run import evaluate_copgen_run


def parse_args():
    p = argparse.ArgumentParser(
        description="Run COP-GEN band-level generation for infilling ablations",
        epilog="Use --list-keys to see all available modality keys from the config."
    )
    p.add_argument("--dataset-root", required=False, type=str,
                   help="MajorTOM dataset root (defaults to <root>/data/cop-gen-small-test)")
    p.add_argument("--model", required=True, type=str,
                   help="Path to Copgen model checkpoint (nnet_ema.pth)")
    p.add_argument("--config", required=True, type=str,
                   help="Path to Copgen model config .py (get_config())")
    p.add_argument("--input-keys", required=True, nargs="+", type=str,
                   help="Exact modality keys to use as conditions (e.g., S2L1C_B02_B03_B04_B08)")
    p.add_argument("--output-keys", required=False, nargs="+", type=str, default=None,
                   help="Exact modality keys to generate. Default: all keys not in --input-keys")
    p.add_argument("--seed", required=False, type=int, default=1234,
                   help="Random seed for reproducibility")
    p.add_argument("--max", dest="max_products", required=False, type=int, default=None,
                   help="Limit number of products")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Batch size for model.generate")
    p.add_argument("--vis-every", type=int, default=0,
                   help="Visualise every N batches (0 to disable)")
    p.add_argument("--visualise-histograms", action="store_true",
                   help="Visualise histograms")
    p.add_argument("--samples", type=int, default=1,
                   help="Number of samples to generate per condition")
    return p.parse_args()


def main():
    args = parse_args()

    gen = CopgenBatchGenerator(
        model_path=Path(args.model),
        config_path=Path(args.config),
        dataset_root=Path(args.dataset_root),
        seed=args.seed,
    )

    # Get all valid modality keys from config
    all_keys = list(gen.model_config.all_modality_configs.keys())

    # Validate input keys
    invalid_inputs = [k for k in args.input_keys if k not in all_keys]
    if invalid_inputs:
        raise ValueError(
            f"Invalid input keys: {invalid_inputs}.\n"
            f"Available keys: {all_keys}"
        )

    # Determine output keys
    if args.output_keys:
        invalid_outputs = [k for k in args.output_keys if k not in all_keys]
        if invalid_outputs:
            raise ValueError(
                f"Invalid output keys: {invalid_outputs}.\n"
                f"Available keys: {all_keys}"
            )
        output_keys = args.output_keys
    else:
        # Default: generate all keys not in input
        output_keys = [k for k in all_keys if k not in args.input_keys]

    # Print summary
    print("=" * 60)
    print("COP-GEN BAND-LEVEL GENERATION (INFILLING ABLATION)")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Config: {args.config}")
    print(f"Dataset root: {args.dataset_root}")
    print(f"Seed: {args.seed}")
    print(f"Batch size: {args.batch_size}")
    print(f"Samples per condition: {args.samples}")
    print(f"Max products: {args.max_products if args.max_products else 'All'}")
    print(f"Visualisation: Every {args.vis_every} batches" if args.vis_every > 0 else "Visualisation: Disabled")
    print(f"\nInput modality keys (conditions):")
    for k in args.input_keys:
        print(f"  - {k}")
    print(f"\nOutput modality keys (to generate):")
    for k in output_keys:
        print(f"  - {k}")
    print("=" * 60)
    print()

    # The keys can be passed directly as "bases" - _modality_keys_for_base() 
    # will return the key itself when it's already a valid key
    gen.process_all(
        condition_bases=args.input_keys,
        generate_bases=output_keys,
        max_products=args.max_products,
        batch_size=args.batch_size,
        vis_every=(args.vis_every if args.vis_every > 0 else None),
        vis_comparison=True,
        visualise_histograms=args.visualise_histograms,
        samples_per_condition=args.samples,
    )

    # Auto-run evaluation after generation
    run_name = f"input_{'_'.join(sorted(args.input_keys))}_output_{'_'.join(sorted(output_keys))}_seed_{args.seed}"
    dataset_root = Path(args.dataset_root)
    run_root = dataset_root / "outputs" / "copgen" / run_name
    metrics = evaluate_copgen_run(dataset_root, run_root, verbose=True)
    print("\nSaved evaluation report to:", run_root / "output_metrics.txt")


if __name__ == "__main__":
    main()

