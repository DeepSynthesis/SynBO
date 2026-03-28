"""
Simple test for ReactionOptimizer initialization and optimization.
Run with: python tests/test_single.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from synbo import ReactionOptimizer
from synbo.utils import load_desc_dict
import pandas as pd


def test_init():
    """Test initialization of ReactionOptimizer."""
    print("=" * 50)
    print("Testing Initialization...")
    print("=" * 50)

    reagent_types = ["base", "ligand", "solvent", "concentration", "temperature"]
    name_suffix = ["_dft", "_dft", "_dft", None, None]
    opt_direct_info = [{"opt_direct": "max", "opt_range": [0, 100]}, {"opt_direct": "min", "opt_range": [0, 0.5]}]

    # Load descriptors
    desc_dict, condition_dict = load_desc_dict(
        reagent_types=reagent_types,
        desc_dir="dataset/descriptors",
        name_suffix=name_suffix,
        return_condition_dict=True,
        index_col=reagent_types,
    )
    print(f"[PASS] Loaded {len(desc_dict)} descriptor files")

    # Create optimizer
    sbo = ReactionOptimizer(opt_metrics=["yield", "cost"], opt_metric_settings=opt_direct_info, opt_type="auto", quiet=True)
    print("[PASS] Created ReactionOptimizer instance")

    # Load reaction space
    sbo.load_rxn_space(condition_dict=condition_dict)
    print("[PASS] Loaded reaction space")

    # Load descriptors
    sbo.load_desc(desc_dict=desc_dict)
    print("[PASS] Loaded descriptors")

    print("[SUCCESS] Initialization test passed!\n")
    return sbo


def test_optimize():
    """Test optimization workflow."""
    print("=" * 50)
    print("Testing Optimization...")
    print("=" * 50)

    reagent_types = ["base", "ligand", "solvent", "concentration", "temperature"]
    name_suffix = ["_dft", "_dft", "_dft", None, None]
    opt_direct_info = [{"opt_direct": "max", "opt_range": [0, 100]}, {"opt_direct": "min", "opt_range": [0, 0.5]}]
    save_dir = "test_output"

    # Load data
    desc_dict, condition_dict = load_desc_dict(
        reagent_types=reagent_types,
        desc_dir="dataset/descriptors",
        name_suffix=name_suffix,
        return_condition_dict=True,
        index_col=reagent_types,
    )

    # Create optimizer
    sbo = ReactionOptimizer(opt_metrics=["yield", "cost"], opt_metric_settings=opt_direct_info, opt_type="auto", quiet=True)
    sbo.load_rxn_space(condition_dict=condition_dict)
    sbo.load_desc(desc_dict=desc_dict)

    # Load previous reaction data
    start_file = Path(__file__).parent / "testfile/start_file.csv"
    if start_file.exists():
        sbo.load_prev_rxn(pd.read_csv(start_file, index_col=False))
        print(f"[PASS] Loaded previous reactions from {start_file}")
    else:
        print(f"[WARN] Start file not found: {start_file}")
        print("[INFO] Running without previous data")

    # Run optimization
    sbo.optimize(
        batch_size=2,
        desc_normalize="minmax",
        refine_desc="auto_select",
        optimize_method="default_BO",
        surrogate_model="RF",
        temperature=0.1,
    )
    print("[PASS] Optimization completed")

    # Save results
    sbo.save_results(save_dir=save_dir, filetype="csv")
    print(f"[PASS] Results saved to {save_dir}")

    print("[SUCCESS] Optimization test passed!\n")
    return sbo


def test_initialize_only():
    """Test initialization with sampling (without optimization)."""
    print("=" * 50)
    print("Testing Initialize Only...")
    print("=" * 50)

    reagent_types = ["base", "ligand", "solvent", "concentration", "temperature"]
    name_suffix = ["_dft", "_dft", "_dft", None, None]
    opt_direct_info = [{"opt_direct": "max", "opt_range": [0, 100]}, {"opt_direct": "min", "opt_range": [0, 0.5]}]

    # Load data
    desc_dict, condition_dict = load_desc_dict(
        reagent_types=reagent_types,
        desc_dir="dataset/descriptors",
        name_suffix=name_suffix,
        return_condition_dict=True,
        index_col=reagent_types,
    )

    # Create optimizer
    sbo = ReactionOptimizer(opt_metrics=["yield", "cost"], opt_metric_settings=opt_direct_info, opt_type="auto", quiet=True)
    sbo.load_rxn_space(condition_dict=condition_dict)
    sbo.load_desc(desc_dict=desc_dict)

    # Initialize (sampling without previous data)
    sbo.initialize(batch_size=5, desc_normalize="minmax", sampling_method="kmeans", refine_desc="filter_0.8")
    print("[PASS] Initialized with kmeans sampling")

    # Save results
    sbo.save_results(save_dir="test_output", filetype="csv")
    print("[PASS] Results saved")

    print("[SUCCESS] Initialize-only test passed!\n")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Running SynBO Single Tests")
    print("=" * 50 + "\n")

    try:
        # Test 1: Basic initialization
        test_init()

        # Test 2: Initialize with sampling
        test_initialize_only()

        # Test 3: Full optimization
        test_optimize()

        print("=" * 50)
        print("ALL TESTS PASSED!")
        print("=" * 50)

    except Exception as e:
        print(f"\n[FAILED] Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
