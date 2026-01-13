from benchmark.generations import TerraMindGenerator
from benchmark.visualize import ComparisonVisualizer
from benchmark.utils import BandStacker
from benchmark.utils.logger import setup_logging
from pathlib import Path
import argparse
import sys
import logging
from utils import set_seed
from benchmark.evaluation.evaluate_terramind_run import evaluate_terramind_run

def parse_args():
    parser = argparse.ArgumentParser(description='Generate and visualize modality transformations (TerraMind)')
    parser.add_argument('--dataset_root', '-d', type=str, default=None,
                       help='Dataset root directory (where Core-*, outputs/, terramind_data/ live)')
    parser.add_argument('--input', '-i', nargs='+', type=str, default=None,
                       help='Input modality(ies) (DEM, LULC, S1RTC, S2L2A, S2L1C, coords). Can specify multiple.')
    parser.add_argument('--inputs-paths', nargs="+", type=str, default=None,
                       help='Optional paths/values for each input modality (order must match --input).')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output modality (DEM, LULC, S1RTC, S2L2A, S2L1C, coords)')
    parser.add_argument('--max_files', '-m', type=int, default=None,
                       help='Maximum number of files to process (default: 10)')
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='Enable visualization of comparisons (default: disabled)')
    parser.add_argument('--seed', type=int, default=1234,
                       help='Random seed to condition TerraMind sampling (default: 1234)')
    parser.add_argument('--experiment_dir', '-x', type=str, default=None,
                       help='Path to an existing TerraMind experiment (outputs/terramind/<experiment_name>) to visualize without regenerating')
    parser.add_argument('--vis_stride', type=int, default=None,
                       help='Visualize every Nth generation (e.g., 128 means take every 128th file)')
    parser.add_argument('--consecutive_visualisations', type=int, default=4,
                       help='Number of consecutive tiles to visualize at each stride interval (default: 4)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    consecutive_visualisations = args.consecutive_visualisations
    
    # Visualize-only mode: user passes an existing experiment directory
    if args.experiment_dir is not None:
        try:
            exp_dir = Path(args.experiment_dir)
            if not exp_dir.exists():
                logging.error(f"Experiment directory does not exist: {exp_dir}")
                sys.exit(1)
            # Expect path: <dataset_root>/outputs/terramind/<experiment_name>
            try:
                DATASET_ROOT = exp_dir.parents[2]
            except IndexError:
                logging.error(f"Invalid experiment path (expected .../outputs/terramind/<experiment_name>): {exp_dir}")
                sys.exit(1)
            experiment_name = exp_dir.name
            if not experiment_name.startswith("input_") or "_output_" not in experiment_name or "_seed_" not in experiment_name:
                logging.error(f"Experiment name does not match expected pattern 'input_<...>_output_<...>_seed_<...>': {experiment_name}")
                sys.exit(1)
            # Parse components from experiment name
            try:
                after_input = experiment_name.split("input_", 1)[1]
                inputs_part, after_output = after_input.split("_output_", 1)
                output_modality, seed_str = after_output.split("_seed_", 1)
                input_modalities = [p for p in inputs_part.split("_") if p]
                parsed_seed = int(seed_str)
            except Exception as e:
                logging.error(f"Failed to parse experiment name '{experiment_name}': {e}")
                sys.exit(1)
            # Set seed from parsed value
            set_seed(parsed_seed)
            # Setup logging within the experiment folder
            log_file = setup_logging(DATASET_ROOT / 'outputs' / 'terramind' / experiment_name / 'logs', experiment_name)
            # Validate modalities
            valid_modalities = ["DEM", "LULC", "S1RTC", "S2L2A", "S2L1C", "coords"]
            for input_mod in input_modalities:
                if input_mod not in valid_modalities:
                    logging.error(f"Invalid input modality '{input_mod}' parsed from experiment name. Must be one of: {valid_modalities}")
                    sys.exit(1)
            if output_modality not in valid_modalities:
                logging.error(f"Invalid output modality '{output_modality}' parsed from experiment name. Must be one of: {valid_modalities}")
                sys.exit(1)
            # Visualize existing generations
            crop_size = 192
            max_files = args.max_files
            logging.info(f"Visualizing existing experiment: {experiment_name}")
            visualizer = ComparisonVisualizer(
                input_modality=input_modalities,
                output_modality=output_modality,
                root=DATASET_ROOT,
                crop_size=crop_size,
                seed=parsed_seed
            )
            visualizer.visualize(n_examples=max_files, stride_every=args.vis_stride, consecutive_visualisations=consecutive_visualisations)
            logging.info("Visualization completed.")
            logging.info(f"Full log saved to: {log_file}")
        except Exception as e:
            logging.error(f"Visualization of existing experiment failed with error: {str(e)}")
            logging.error(f"Error type: {type(e).__name__}")
            import traceback
            logging.error(f"Full traceback:\n{traceback.format_exc()}")
            raise
        sys.exit(0)
    
    # Standard generation/evaluation/visualization flow
    if args.dataset_root is None or args.input is None or args.output is None:
        logging.error("In generation mode, --dataset_root, --input and --output are required (or use --experiment_dir for visualize-only).")
        sys.exit(1)
    
    DATASET_ROOT = Path(args.dataset_root)
    
    # Set global RNG for reproducible/stochastic sampling per seed
    set_seed(int(args.seed))
    
    experiment_name = f"input_{'_'.join(args.input)}_output_{args.output}_seed_{int(args.seed)}"
    log_file = setup_logging(DATASET_ROOT / 'outputs' / 'terramind' / experiment_name / 'logs', experiment_name)
    
    try:
        valid_modalities = ["DEM", "LULC", "S1RTC", "S2L2A", "S2L1C", "coords"]
        
        for input_mod in args.input:
            if input_mod not in valid_modalities:
                logging.error(f"Invalid input modality '{input_mod}'. Must be one of: {valid_modalities}")
                sys.exit(1)
        
        if args.output not in valid_modalities:
            logging.error(f"Invalid output modality '{args.output}'. Must be one of: {valid_modalities}")
            sys.exit(1)
        
        input_modalities = args.input
        output_modality = args.output

        input_overrides = None
        if args.inputs_paths:
            if len(args.inputs_paths) != len(args.input):
                logging.error(f"--inputs-paths expects exactly {len(args.input)} entries. Got {len(args.inputs_paths)}.")
                sys.exit(1)
            input_overrides = {mod: path for mod, path in zip(args.input, args.inputs_paths)}
            logging.info(f"Using input overrides: {input_overrides}")

        max_stacking_files = None
        max_files = args.max_files
        crop_size = 192

        logging.info(f"Processing: {' + '.join(input_modalities)} → {args.output}")

        # Step 0: Stack bands for all input modalities
        # for modality in input_modalities:
        for modality in ["DEM", "LULC", "S1RTC", "S2L2A", "S2L1C", "coords"]:
            dir_path = DATASET_ROOT / "terramind_inputs" / modality
            tif_files = list(dir_path.glob("*.tif")) if dir_path.exists() else []
            if not dir_path.exists() or len(tif_files) == 0:
                logging.info(f"Stacking bands for {modality}")
                BandStacker(modality=modality, root=DATASET_ROOT).stack_all()
            else:
                logging.info(f"Bands for {modality} already stacked ({len(tif_files)} files).")

        # Step 1: Generate outputs
        output_dir = DATASET_ROOT / "outputs" / "terramind" / experiment_name / "generations"
        tif_files = list(output_dir.glob("*.tif")) if output_dir.exists() else []
        # if not output_dir.exists() or len(tif_files) == 0:
        logging.info(f"Generating {output_modality} from {input_modalities} (max {max_files} files)...")
        generator = TerraMindGenerator(
            input_modalities=input_modalities,
            output_modality=output_modality,
            model_name="terramind_v1_base_generate",
            crop_size=crop_size,
            timesteps=50,
            pretrained=True,
            standardize=True,
            device="cuda",
            root=DATASET_ROOT,
            seed=args.seed,
            input_overrides=input_overrides
        )
        generator.process_all(max_files=max_files)
        # else:
        #     logging.info(f"Outputs for {output_modality} already generated ({len(tif_files)} files).")
        
        # Step 2: Evaluate outputs
        evaluate_terramind_run(DATASET_ROOT, DATASET_ROOT / "outputs" / "terramind" / experiment_name, output_modality, verbose=True)
        logging.info(f"Evaluation report saved to: {output_dir / 'output_metrics.txt'}")

        # Step 2: Visualize comparisons (if enabled)
        if args.visualize:
            logging.info("Starting visualization...")
            visualizer = ComparisonVisualizer(
                input_modality=input_modalities,  
                output_modality=output_modality,
                root=DATASET_ROOT,
                crop_size=crop_size,
                seed=args.seed
            )
            visualizer.visualize(n_examples=max_files, stride_every=args.vis_stride, consecutive_visualisations=consecutive_visualisations)
        else:
            logging.info("Visualization skipped (use --visualize to enable)")
        
        logging.info(f"Experiment completed successfully!")
        logging.info(f"Full log saved to: {log_file}")

    except Exception as e: 
        logging.error(f"Experiment failed with error: {str(e)}")
        logging.error(f"Error type: {type(e).__name__}")
        
        import traceback
        logging.error(f"Full traceback:\n{traceback.format_exc()}")
        
        raise
