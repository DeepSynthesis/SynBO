import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import glob

from utils.plots_batch import (
    plot_optimization_curves,
    plot_hypervolume_coverage,
    plot_final_distribution_boxplot,
    plot_optimization_process_scatter,
)


def plot_comparison(
    model_results: Dict[str, Dict],
    output_dir: str,
    full_space_file: Optional[str] = None,
    plot_types: List[str] = ["curves", "hv", "boxplot", "scatter"],
):
    """
    Plot comparison results for multiple models

    Args:
        model_results: Model results dictionary, format:
            {
                "model_name": {
                    "results_path": "path/to/results.csv",  # Support wildcards，such as "results/multiple_*/all_batches_final_round_*.csv"
                    "target_columns": ["yield", "cost"],    # Target value columns to evaluate
                    "direction_tags": ["max", "min"],       # Optimization direction: "max" or "min"
                },
                ...
            }
        output_dir: Output directory
        full_space_file: Full space data file paths(for HV and Pareto front calculation)
        plot_types: List of plot types to draw，optional: "curves", "hv", "boxplot", "scatter"

    Returns:
        None (Save plots to output_dir)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*20} Starting Comparison Plotting {'='*20}")
    print(f"Output Directory: {output_dir}")

    # Load data for all models
    all_model_data = {}
    for model_name, model_config in model_results.items():
        results_path = model_config["results_path"]

        # Use glob to find matching files
        matching_files = glob.glob(results_path)

        if not matching_files:
            print(f"Warning: No files found for {model_name} with pattern: {results_path}")
            continue

        print(f"Found {len(matching_files)} files for {model_name}")

        # Store all runs for this model
        model_runs = []

        for file_path in matching_files:

            df = pd.read_csv(file_path)

            model_runs.append(df)

        if not model_runs:
            print(f"Warning: No valid data loaded for {model_name}")
            continue

        # Store to all_model_data
        all_model_data[model_name] = model_runs
        print(f"Loaded {len(model_runs)} runs for {model_name}")

    if not all_model_data:
        print("Error: No valid model data found to plot.")
        return

    print(f"\nLoaded {len(all_model_data)} models for comparison")

    # Set Seaborn style
    sns.set_theme(style="whitegrid")

    # Get config info
    first_model = list(all_model_data.keys())[0]
    model_config = model_results[first_model]
    target_columns = model_config["target_columns"]
    direction_tags = model_config["direction_tags"]

    print(f"\n{'-'*20} Generating combined comparison plots {'-'*20}")

    # 1. Plot optimization curves(all models on one figure, with confidence intervals)
    if "curves" in plot_types:
        plot_optimization_curves(all_model_data, target_columns, direction_tags, output_dir)

    # 2. Plot hypervolume coverage
    if "hv" in plot_types and full_space_file and Path(full_space_file).exists():
        plot_hypervolume_coverage(all_model_data, target_columns, direction_tags, Path(full_space_file), output_dir)

    # 3. Plot final best distribution
    if "boxplot" in plot_types:
        plot_final_distribution_boxplot(all_model_data, target_columns, direction_tags, output_dir)

    # 4. Plot optimization process scatter(Pareto front comparison)(all models on one figure)
    if "scatter" in plot_types and full_space_file and Path(full_space_file).exists():
        plot_optimization_process_scatter(all_model_data, target_columns, direction_tags, Path(full_space_file), output_dir)

    print(f"\n{'='*20} All comparison plots completed {'='*20}")
    print(f"Results saved to: {output_dir}")


# Example usage
if __name__ == "__main__":
    # Example: compare synbo and EDBOplus results
    # Each model results from multiple CSV files，each file represents an independent run
    model_results = {
        "SynBO": {
            "results_path": "results/multiple_20260421_135123/all_batches_final_round_*.csv",
            "target_columns": ["yield", "cost"],
            "direction_tags": ["max", "min"],
        },
        "EDBOplus": {
            "results_path": "compare_mothods/edboplus/results/EDBOplus_for_B-H_HTE/batch_*.csv",
            "target_columns": ["yield", "cost"],
            "direction_tags": ["max", "min"],
        },
        "Gryffin": {
            "results_path": "compare_mothods/gryffin/results/B-H_HTE/batch_*.csv",
            "target_columns": ["yield", "cost"],
            "direction_tags": ["max", "min"],
        },
        "Random": {
            "results_path": "compare_mothods/random/results/B-H_HTE/batch_*.csv",
            "target_columns": ["yield", "cost"],
            "direction_tags": ["max", "min"],
        },
    }

    full_space_file = "datasets/HTE_datasets/B-H_HTE/B-H_HTE.csv"
    output_dir = "comparison_results/B-H_HTE"

    plot_comparison(
        model_results=model_results,
        output_dir=output_dir,
        full_space_file=full_space_file,
        plot_types=["curves", "hv", "boxplot", "scatter"],
    )
