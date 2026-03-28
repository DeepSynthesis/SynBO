"""Example demonstrating hypervolume calculation functionality."""

import pandas as pd
from synbo import ReactionOptimizer


# Example 1: Calculate HV for current optimization progress
def example_current_hv():
    """Calculate hypervolume for current batch."""
    # Initialize optimizer
    sbo = ReactionOptimizer(
        opt_metrics=["yield", "ee"],
        opt_metric_settings=[{"opt_direct": "max", "opt_range": [0, 100]}, {"opt_direct": "max", "opt_range": [0, 100]}],
        quiet=False,
    )

    # Load reaction space (simplified example)
    condition_dict = {"solvent": ["toluene", "DMF", "THF"], "base": ["NaOH", "K2CO3", "Et3N"], "temperature": [25, 50, 75]}
    sbo.load_rxn_space(condition_dict)

    # Create sample previous reaction data
    prev_data = pd.DataFrame(
        {
            "solvent": ["toluene", "DMF", "THF", "toluene", "DMF"],
            "base": ["NaOH", "K2CO3", "Et3N", "K2CO3", "NaOH"],
            "temperature": [25, 50, 75, 50, 25],
            "yield": [45.2, 67.8, 52.3, 71.5, 58.9],
            "ee": [78.5, 85.2, 72.1, 88.7, 80.3],
            "batch": [0, 0, 0, 1, 1],
        }
    )

    # Load previous reaction data
    sbo.load_prev_rxn(prev_data)

    # Calculate hypervolume for current state (all batches)
    hv_result = sbo.calculate_current_hv()
    print("\n=== Current Hypervolume ===")
    print(f"HV: {hv_result['hv']:.4f}")
    print(f"HV (normalized): {hv_result['hv_normalized']:.4f}")
    print(f"Total HV: {hv_result['total_hv']:.4f}")
    print(f"Number of points: {hv_result['num_points']}")
    print(f"Batch ID: {hv_result['batch_id']}")

    # Calculate hypervolume up to a specific batch
    hv_batch0 = sbo.calculate_current_hv(batch_id=0)
    print(f"\n=== Hypervolume up to batch 0 ===")
    print(f"HV: {hv_batch0['hv']:.4f}")
    print(f"HV (normalized): {hv_batch0['hv_normalized']:.4f}")


def example_hv_by_batch():
    """Calculate hypervolume for each batch cumulatively."""
    # Initialize optimizer
    sbo = ReactionOptimizer(
        opt_metrics=["yield", "cost"],
        opt_metric_settings=[{"opt_direct": "max", "opt_range": [0, 100]}, {"opt_direct": "min", "opt_range": [0, 1000]}],
        quiet=False,
    )

    # Load reaction space
    condition_dict = {"solvent": ["toluene", "DMF", "THF"], "catalyst": ["Pd", "Ni", "Cu"]}
    sbo.load_rxn_space(condition_dict)

    # Create sample data with multiple batches
    prev_data = pd.DataFrame(
        {
            "solvent": ["toluene", "DMF", "THF", "toluene", "DMF", "THF", "toluene"],
            "catalyst": ["Pd", "Ni", "Cu", "Ni", "Pd", "Pd", "Cu"],
            "yield": [45.2, 67.8, 52.3, 71.5, 78.2, 65.4, 82.1],
            "cost": [500, 300, 200, 320, 550, 280, 220],
            "batch": [0, 0, 0, 1, 1, 2, 2],
        }
    )

    # Load previous reaction data
    sbo.load_prev_rxn(prev_data)

    # Calculate hypervolume for each batch
    hv_by_batch = sbo.calculate_hv_by_batch()
    print("\n=== Hypervolume by Batch ===")
    print(hv_by_batch)

    # You can also plot the hypervolume progress
    print("\nHypervolume Progress:")
    for idx, row in hv_by_batch.iterrows():
        print(f"Batch {row['batch']}: HV = {row['hv']:.4f} (normalized: {row['hv_normalized']:.4f})")


def example_direct_function_call():
    """Example of calling HV calculation functions directly."""
    from synbo.utils.hv_calculator import calculate_hypervolume_for_batch, calculate_hypervolume_by_batch

    # Create sample data
    prev_data = pd.DataFrame({"yield": [45.2, 67.8, 52.3, 71.5, 78.2], "ee": [78.5, 85.2, 72.1, 88.7, 82.3], "batch": [0, 0, 0, 1, 1]})

    opt_metrics = ["yield", "ee"]
    opt_metric_settings = [{"opt_direct": "max", "opt_range": [0, 100]}, {"opt_direct": "max", "opt_range": [0, 100]}]

    # Calculate hypervolume directly
    hv_result = calculate_hypervolume_for_batch(
        prev_rxn_info=prev_data,
        opt_metrics=opt_metrics,
        opt_metric_settings=opt_metric_settings,
        batch_id=1,
        reference_point_multiplier=1.1,
    )

    print("\n=== Direct Function Call ===")
    print(f"HV: {hv_result['hv']:.4f}")
    print(f"HV (normalized): {hv_result['hv_normalized']:.4f}")

    # Calculate by batch
    hv_by_batch = calculate_hypervolume_by_batch(prev_rxn_info=prev_data, opt_metrics=opt_metrics, opt_metric_settings=opt_metric_settings)
    print("\n=== HV by Batch (Direct Call) ===")
    print(hv_by_batch)


if __name__ == "__main__":
    print("=" * 60)
    print("Example 1: Calculate Current Hypervolume")
    print("=" * 60)
    example_current_hv()

    print("\n" + "=" * 60)
    print("Example 2: Calculate Hypervolume by Batch")
    print("=" * 60)
    example_hv_by_batch()

    print("\n" + "=" * 60)
    print("Example 3: Direct Function Call")
    print("=" * 60)
    example_direct_function_call()
