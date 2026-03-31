#!/usr/bin/env python3
"""
Optimize script for SynBO reaction optimization.

This script demonstrates how to run Bayesian optimization with previous
reaction data to recommend new experimental conditions.

Usage:
    python optimize.py --desc-dir dataset/descriptors --input testfile/start_file.csv --output output/optimize
"""

import argparse
import json
from pathlib import Path

from synbo import ReactionOptimizer
from synbo.utils import load_desc_dict
import pandas as pd


def load_optimization_settings():
    """Load optimization settings from config.json and optimization_settings.json."""
    # Read config.json to get project_wd
    config_path = Path(__file__).parent.parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found at {config_path}. Please configure project first.")
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    project_wd = Path(config.get("project_wd"))
    if not project_wd or not project_wd.exists():
        raise ValueError(f"Invalid project_wd in config.json: {project_wd}")
    
    # Read optimization_settings.json
    opt_settings_path = project_wd / "optimization_settings.json"
    if not opt_settings_path.exists():
        raise FileNotFoundError(
            f"optimization_settings.json not found at {opt_settings_path}. "
            "Please define optimization metrics first."
        )
    
    with open(opt_settings_path, "r") as f:
        opt_settings = json.load(f)
    
    settings = opt_settings.get("optimization_settings", {})
    opt_metrics = settings.get("opt_metrics", [])
    opt_direct_info = settings.get("opt_direct_info", [])
    
    if not opt_metrics:
        raise ValueError("No optimization metrics found in optimization_settings.json")
    
    return opt_metrics, opt_direct_info


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Bayesian optimization with previous reaction data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Optimize with default settings
  python optimize.py --desc-dir dataset/descriptors --input testfile/start_file.csv --output output/optimize

  # Optimize with custom batch size
  python optimize.py --desc-dir dataset/descriptors --input testfile/start_file.csv \\
                     --output output/optimize --batch-size 5

  # Optimize with custom configuration
  python optimize.py --desc-dir dataset/descriptors --input testfile/start_file.csv \\
                     --output output/optimize --reagent-types base ligand solvent \\
                     --surrogate-model RF --temperature 0.2
        """,
    )

    parser.add_argument(
        "--desc-dir",
        type=Path,
        required=True,
        help="Directory containing descriptor files",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to CSV file with previous reaction data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory for results (default: output/optimize)",
    )
    parser.add_argument(
        "--reagent-types",
        nargs="+",
        help="List of reagent types",
    )
    parser.add_argument(
        "--name-suffix",
        nargs="+",
        help="Name suffixes for descriptor files",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of new conditions to recommend",
    )
    parser.add_argument(
        "--desc-normalize",
        default="zscore",
        choices=["minmax", "zscore", "l2"],
        help="Descriptor normalization method (default: zscore)",
    )
    parser.add_argument(
        "--refine-desc",
        default="pass",
        choices=["auto_select", "filter_only", "pass"],
        help="Descriptor refinement method (default: pass)",
    )
    parser.add_argument(
        "--optimize-method",
        default="default_BO",
        help="Optimization algorithm to use (default: default_BO)",
    )
    parser.add_argument(
        "--surrogate-model",
        default="GP",
        help="Surrogate model type (default: GP)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    return parser.parse_args()


def main() -> int:
    """Main function to run optimization workflow."""
    args = parse_arguments()

    name_suffix = [s if s != "None" else None for s in args.name_suffix]

    # Load optimization settings from configuration files
    print("Step 1: Loading optimization settings...")
    print("-" * 60)
    try:
        opt_metrics, opt_direct_info = load_optimization_settings()
        print(f"✓ Loaded optimization metrics: {opt_metrics}")
        print(f"  Metrics configuration: {len(opt_direct_info)} metric(s)")
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        return 1
    print()

    print("Step 2: Loading descriptors...")
    print("-" * 60)

    # Load descriptors
    desc_dict, condition_dict = load_desc_dict(
        reagent_types=args.reagent_types,
        desc_dir=str(args.desc_dir),
        name_suffix=name_suffix,
        return_condition_dict=True,
        index_col=args.reagent_types,
    )
    print(f"✓ Loaded {len(desc_dict)} descriptor files")
    print()

    print("Step 3: Creating ReactionOptimizer instance...")
    print("-" * 60)

    # Create optimizer
    sbo = ReactionOptimizer(
        opt_metrics=opt_metrics,
        opt_metric_settings=opt_direct_info,
        opt_type="auto",
        random_seed=args.random_seed,
        quiet=args.quiet,
    )
    print("✓ ReactionOptimizer instance created")
    print()

    print("Step 4: Loading reaction space...")
    print("-" * 60)

    # Load reaction space
    sbo.load_rxn_space(condition_dict=condition_dict)
    print("✓ Reaction space loaded")
    print()

    print("Step 5: Loading descriptors...")
    print("-" * 60)

    # Load descriptors
    sbo.load_desc(desc_dict=desc_dict)
    print("✓ Descriptors loaded")
    print()

    print("Step 6: Loading previous reaction data...")
    print("-" * 60)

    # Load previous reaction data
    if not args.input.exists():
        print(f"[ERROR] Input file not found: {args.input}")
        print()
        print("Please provide previous reaction data to run optimization.")
        print("If you want to start with initial sampling, use initialize.py instead.")
        return 1

    prev_rxn_data = pd.read_csv(args.input, index_col=False)
    sbo.load_prev_rxn(prev_rxn_data)
    print(f"✓ Loaded previous reactions from {args.input}")
    print(f"  Total reactions: {len(prev_rxn_data)}")
    if "batch" in prev_rxn_data.columns:
        print(f"  Batch range: {prev_rxn_data['batch'].min()} to {prev_rxn_data['batch'].max()}")
    print()

    print("Step 7: Running optimization...")
    print("-" * 60)

    # Run optimization
    sbo.optimize(
        batch_size=args.batch_size,
        desc_normalize=args.desc_normalize,
        refine_desc=args.refine_desc,
        optimize_method=args.optimize_method,
        surrogate_model=args.surrogate_model,
        temperature=args.temperature,
    )
    print()

    print("Step 8: Saving results...")
    print("-" * 60)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Save results
    sbo.save_results(save_dir=str(args.output), filetype="csv")
    sbo.save_results(save_dir=str(args.output), filetype="excel")
    print(f"✓ Results saved to {args.output}")
    print()

    # Display optimization summary
    print("=" * 60)
    print("Optimization completed successfully!")
    print("=" * 60)
    print()
    print("Summary:")
    print(f"  - Recommended {len(sbo.selected_conditions)} new conditions")
    print(f"  - Exploit: {sum(1 for t in sbo.recommend_type if t == 'exploit')}")
    print(f"  - Explore: {sum(1 for t in sbo.recommend_type if t == 'explore')}")
    print()
    print("Next steps:")
    print("1. Run the experiments with the recommended conditions")
    print("2. Collect the experimental results")
    print("3. Update your reaction data file")
    print("4. Run optimize.py again to continue optimization")
    print()

    return 0


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] Optimization failed: {e}")
