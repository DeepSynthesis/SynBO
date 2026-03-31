#!/usr/bin/env python3
"""
Initialize script for SynBO reaction optimization.

This script demonstrates how to initialize the ReactionOptimizer and perform
initial sampling without previous reaction data.

Usage:
    python initialize.py --desc-dir dataset/descriptors --output output/initialize
"""

import argparse
import json
import sys
from pathlib import Path

from synbo import ReactionOptimizer
from synbo.utils import load_desc_dict


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
        description="Initialize reaction optimization with initial sampling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize with default settings
  python initialize.py --desc-dir dataset/descriptors --output output/initialize

  # Initialize with custom batch size and sampling method
  python initialize.py --desc-dir dataset/descriptors --output output/initialize \\
                       --batch-size 10 --sampling-method sobol

  # Initialize with custom configuration
  python initialize.py --desc-dir dataset/descriptors --output output/initialize \\
                       --reagent-types base ligand solvent --batch-size 5
        """,
    )

    parser.add_argument(
        "--desc-dir",
        type=Path,
        required=True,
        help="Directory containing descriptor files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/initialize"),
        help="Output directory for results (default: output/initialize)",
    )
    parser.add_argument(
        "--reagent-types",
        nargs="+",
        default=["base", "ligand", "solvent", "concentration", "temperature"],
        help="List of reagent types (default: base ligand solvent concentration temperature)",
    )
    parser.add_argument(
        "--name-suffix",
        nargs="+",
        default=["_dft", "_dft", "_dft", None, None],
        help="Name suffixes for descriptor files (default: _dft _dft _dft None None)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of initial samples to generate (default: 5)",
    )
    parser.add_argument(
        "--desc-normalize",
        default="minmax",
        choices=["minmax", "zscore", "l2"],
        help="Descriptor normalization method (default: minmax)",
    )
    parser.add_argument(
        "--sampling-method",
        default="kmeans",
        choices=["sobol", "random", "lhs", "kmeans"],
        help="Sampling strategy for initial points (default: kmeans)",
    )
    parser.add_argument(
        "--refine-desc",
        default="filter_0.8",
        choices=["auto_select", "filter_only", "pass", "filter_0.8"],
        help="Descriptor refinement method (default: filter_0.8)",
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
    """Main function to run initialization workflow."""
    args = parse_arguments()

    print("=" * 60)
    print("SynBO Initialization Script")
    print("=" * 60)
    print()

    # Handle None values in name_suffix
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
        opt_type="init",
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

    print("Step 6: Running initialization with sampling...")
    print("-" * 60)

    # Initialize with sampling
    sbo.initialize(
        batch_size=args.batch_size,
        desc_normalize=args.desc_normalize,
        sampling_method=args.sampling_method,
        refine_desc=args.refine_desc,
    )
    print()

    print("Step 7: Saving results...")
    print("-" * 60)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Save results
    sbo.save_results(save_dir=str(args.output), filetype="csv")
    sbo.save_results(save_dir=str(args.output), filetype="excel")
    print(f"✓ Results saved to {args.output}")
    print()

    print("=" * 60)
    print("Initialization completed successfully!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Run the experiments with the recommended conditions")
    print("2. Collect the experimental results")
    print("3. Use optimize.py to continue optimization with the new data")
    print()

    return 0


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] Initialization failed: {e}")
