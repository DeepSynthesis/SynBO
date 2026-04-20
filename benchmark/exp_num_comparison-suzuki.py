"""
Compare experiment efficiency between SynBO, OFAT, and Random methods.

This script generates a plot showing the expected target value (e.g., Conversion)
versus the number of experiments conducted.

- SynBO: Reads CSV files from multiple runs, plots cumulative best values per batch (5 experiments per batch)
- OFAT: Reads from ofat_expected_results.json
- Random: Reads from random_expected_results.json
"""

import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple


def load_synbo_data(results_pattern: str, target_column: str = "Conversion", direction: str = "max") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load SynBO data from CSV files and compute cumulative best values.

    Args:
        results_pattern: Glob pattern to find CSV files (e.g., "results/multiple_*/all_batches_final_round_*.csv")
        target_column: Name of the target column
        direction: Optimization direction ("max" or "min")

    Returns:
        Tuple of (experiment_numbers, mean_best_values)
    """
    # Find all matching files
    matching_files = glob.glob(results_pattern)

    if not matching_files:
        print(f"Warning: No files found with pattern: {results_pattern}")
        return np.array([]), np.array([])

    print(f"Found {len(matching_files)} total files, filtering for '{target_column}' column...")

    # Store cumulative best trajectories from all runs
    all_trajectories = []
    max_batches = 0
    valid_files = 0

    for file_path in matching_files:
        df = pd.read_csv(file_path)

        # Skip files that don't have the target column (different experiment type)
        if target_column not in df.columns:
            continue

        valid_files += 1

        # Sort by batch
        df = df.sort_values("batch").copy()

        # Group by batch and get best value per batch
        if direction == "max":
            batch_best = df.groupby("batch")[target_column].max()
        elif direction == "min":
            batch_best = df.groupby("batch")[target_column].min()
        else:
            raise ValueError(f"Unknown direction '{direction}'. Use 'max' or 'min'.")

        # Calculate cumulative best
        if direction == "max":
            cumulative_best = batch_best.cummax()
        else:
            cumulative_best = batch_best.cummin()

        # Store trajectory
        trajectory = cumulative_best.values
        all_trajectories.append(trajectory)
        max_batches = max(max_batches, len(trajectory))

    print(f"Found {valid_files} valid SynBO runs with '{target_column}' column")

    if not all_trajectories:
        return np.array([]), np.array([])

    # Pad trajectories to same length and compute mean
    # Each batch has 5 experiments
    experiments_per_batch = 5

    padded_trajectories = []
    for traj in all_trajectories:
        if len(traj) < max_batches:
            # Pad with last value
            padded = np.pad(traj, (0, max_batches - len(traj)), mode='edge')
        else:
            padded = traj
        padded_trajectories.append(padded)

    # Convert to array: shape (n_runs, n_batches)
    trajectories_array = np.array(padded_trajectories)

    # Compute mean and std across runs
    mean_best_values = trajectories_array.mean(axis=0)

    # Experiment numbers: each batch corresponds to 5 experiments
    # Batch 0 is the first batch (5 experiments), Batch 1 is next (10 experiments total), etc.
    experiment_numbers = np.arange(1, max_batches + 1) * experiments_per_batch

    return experiment_numbers, mean_best_values

def load_baseline_data(json_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load baseline method (OFAT or Random) data from JSON file.

    The JSON format is:
    {
        "90": {"threshold": 90, "mean": 30.399, ...},
        "91": {"threshold": 91, "mean": 34.464, ...},
        ...
    }

    Args:
        json_path: Path to the JSON file

    Returns:
        Tuple of (mean_experiment_numbers, thresholds)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    thresholds = []
    mean_experiments = []

    for key, value in data.items():
        thresholds.append(float(value["threshold"]))
        mean_experiments.append(float(value["mean"]))

    # Sort by threshold
    sorted_indices = np.argsort(thresholds)
    thresholds = np.array(thresholds)[sorted_indices]
    mean_experiments = np.array(mean_experiments)[sorted_indices]

    return mean_experiments, thresholds


def plot_experiment_comparison(
    synbo_pattern: str,
    ofat_json: str,
    random_json: str,
    output_dir: str = "comparison_results/exp_num_comparison",
    target_column: str = "Conversion",
    direction: str = "max",
):
    """
    Generate comparison plot of experiment efficiency.

    Args:
        synbo_pattern: Glob pattern for SynBO CSV files
        ofat_json: Path to OFAT results JSON
        random_json: Path to Random results JSON
        output_dir: Output directory for the plot
        target_column: Name of the target column
        direction: Optimization direction ("max" or "min")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*20} Starting Experiment Number Comparison {'='*20}")

    # Load data for all methods
    print("\nLoading SynBO data...")
    synbo_exp_nums, synbo_values = load_synbo_data(synbo_pattern, target_column, direction)

    print("\nLoading OFAT data...")
    ofat_exp_nums, ofat_values = load_baseline_data(ofat_json)

    print("\nLoading Random data...")
    random_exp_nums, random_values = load_baseline_data(random_json)

    # Set style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 7))

    # Plot SynBO
    if len(synbo_exp_nums) > 0:
        plt.plot(
            synbo_exp_nums,
            synbo_values,
            marker='o',
            markersize=6,
            linewidth=2.5,
            label='SynBO',
            color='#2E86AB',
            zorder=10,
        )
        print(f"SynBO: {len(synbo_exp_nums)} batches plotted")

    # Plot OFAT
    if len(ofat_exp_nums) > 0:
        plt.plot(
            ofat_exp_nums,
            ofat_values,
            marker='s',
            markersize=6,
            linewidth=2.5,
            label='OFAT',
            color='#A23B72',
            zorder=9,
        )
        print(f"OFAT: {len(ofat_exp_nums)} thresholds plotted")

    # Plot Random
    if len(random_exp_nums) > 0:
        plt.plot(
            random_exp_nums,
            random_values,
            marker='^',
            markersize=6,
            linewidth=2.5,
            label='Random',
            color='#F18F01',
            zorder=8,
        )
        print(f"Random: {len(random_exp_nums)} thresholds plotted")

    # Formatting
    plt.xlabel("Number of Experiments", fontsize=14, fontname="Arial", fontweight="bold")
    plt.ylabel(f"Expected {target_column}", fontsize=14, fontname="Arial", fontweight="bold")
    plt.title(f"Experiment Efficiency Comparison: {target_column}", fontsize=16, fontname="Arial", fontweight="bold")

    plt.tick_params(axis="both", which="major", labelsize=12, width=1.5, length=6)
    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        label.set_fontname("Arial")

    plt.grid(True, linestyle="--", alpha=0.6, linewidth=0.8)
    ax = plt.gca()
    ax.spines["top"].set_linewidth(1.2)
    ax.spines["right"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["left"].set_linewidth(1.2)

    plt.legend(loc="best", fontsize=12, framealpha=0.9)

    # Save plot
    save_path = output_dir / f"exp_num_comparison_{target_column.lower()}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved: {save_path}")

    # Print summary statistics
    print(f"\n{'='*20} Summary Statistics {'='*20}")
    if len(synbo_exp_nums) > 0:
        print(f"SynBO final: {synbo_values[-1]:.2f} at {synbo_exp_nums[-1]} experiments")
    if len(ofat_exp_nums) > 0:
        print(f"OFAT max threshold: {ofat_values[-1]:.2f} at {ofat_exp_nums[-1]:.1f} experiments (on average)")
    if len(random_exp_nums) > 0:
        print(f"Random max threshold: {random_values[-1]:.2f} at {random_exp_nums[-1]:.1f} experiments (on average)")

    # Calculate efficiency gains
    print(f"\n{'='*20} Efficiency Analysis {'='*20}")
    if len(synbo_exp_nums) > 0 and len(ofat_exp_nums) > 0 and len(random_exp_nums) > 0:
        # Compare at common target values (e.g., 90, 92, 94, 95)
        comparison_thresholds = [90, 92, 94, 95]

        for threshold in comparison_thresholds:
            # Check if SynBO reaches this threshold
            synbo_idx = np.where(synbo_values >= threshold)[0]
            if len(synbo_idx) > 0:
                synbo_exp_needed = synbo_exp_nums[synbo_idx[0]]

                # Find OFAT experiments needed for this threshold
                ofat_idx = np.where(ofat_values >= threshold)[0]
                if len(ofat_idx) > 0:
                    ofat_exp_needed = ofat_exp_nums[ofat_idx[0]]
                else:
                    ofat_exp_needed = None

                # Find Random experiments needed for this threshold
                random_idx = np.where(random_values >= threshold)[0]
                if len(random_idx) > 0:
                    random_exp_needed = random_exp_nums[random_idx[0]]
                else:
                    random_exp_needed = None

                print(f"\nTo achieve {target_column} ≥ {threshold}:")
                print(f"  - SynBO needs: {synbo_exp_needed:.0f} experiments")

                if ofat_exp_needed:
                    savings = (ofat_exp_needed - synbo_exp_needed) / ofat_exp_needed * 100
                    print(f"  - OFAT needs: {ofat_exp_needed:.1f} experiments (on average)")
                    print(f"  - SynBO saves {savings:.1f}% experiments vs OFAT")
                else:
                    print(f"  - OFAT: threshold not reached")

                if random_exp_needed:
                    savings = (random_exp_needed - synbo_exp_needed) / random_exp_needed * 100
                    print(f"  - Random needs: {random_exp_needed:.1f} experiments (on average)")
                    print(f"  - SynBO saves {savings:.1f}% experiments vs Random")
                else:
                    print(f"  - Random: threshold not reached")

        # Also show final comparison at SynBO's final value
        synbo_final_value = synbo_values[-1]
        synbo_final_exp = synbo_exp_nums[-1]

        # Find the closest threshold in OFAT and Random data
        ofat_idx = np.argmin(np.abs(ofat_values - synbo_final_value))
        random_idx = np.argmin(np.abs(random_values - synbo_final_value))

        print(f"\n{'='*20} Summary at SynBO's Final Performance {'='*20}")
        print(f"SynBO achieves {synbo_final_value:.2f} {target_column} at {synbo_final_exp:.0f} experiments")
        print(f"OFAT needs {ofat_exp_nums[ofat_idx]:.1f} experiments to reach {ofat_values[ofat_idx]:.2f} (on average)")
        print(f"Random needs {random_exp_nums[random_idx]:.1f} experiments to reach {random_values[random_idx]:.2f} (on average)")

    print(f"\n{'='*20} Comparison Complete {'='*20}")


if __name__ == "__main__":
    # Configuration
    synbo_pattern = "results/multiple_*/all_batches_final_round_*.csv"
    ofat_json = "compare_mothods/ofat/results/ofat_expected_results.json"
    random_json = "compare_mothods/random/results/random_expected_results.json"
    output_dir = "comparison_results/exp_num_comparison"

    # Generate comparison plot
    plot_experiment_comparison(
        synbo_pattern=synbo_pattern,
        ofat_json=ofat_json,
        random_json=random_json,
        output_dir=output_dir,
        target_column="Conversion",
        direction="max",
    )
