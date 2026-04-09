import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional

from utils.plots import (
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
                    "results_path": "path/to/results.csv",  # results.csvFile paths
                    "target_columns": ["yield", "cost"],    # Target value columns to evaluate
                    "direction_tags": ["max", "min"],       # Optimization direction: "max" or "min"
                    "range_tags": [[0, 100], [0, 0.1]],     # Range of target values
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
        results_path = Path(model_config["results_path"])
        if not results_path.exists():
            print(f"Warning: Results file not found for {model_name}: {results_path}")
            continue

        # Read result file
        df = pd.read_csv(results_path)

        # Standardize column names：Unify different column formats to yield and cost
        # EDBOplus uses yield_collected_values
        if "yield_collected_values" in df.columns and "yield" not in df.columns:
            df = df.rename(columns={"yield_collected_values": "yield"})
        if "cost_collected_values" in df.columns and "cost" not in df.columns:
            df = df.rename(columns={"cost_collected_values": "cost"})

        # Update target_columns to standardized names
        model_config["target_columns"] = [
            "yield" if col == "yield_collected_values" else "cost" if col == "cost_collected_values" else col
            for col in model_config["target_columns"]
        ]

        # May need from multiple round files
        if "round_index" in df.columns:
            # Already merged data
            all_model_data[model_name] = [df]
        else:
            # single round data
            df["round_index"] = 0
            all_model_data[model_name] = [df]

    if not all_model_data:
        print("Error: No valid model data found to plot.")
        return

    print(f"Loaded {len(all_model_data)} models for comparison")

    # Set Seaborn style
    sns.set_theme(style="whitegrid")

    # Get config info
    first_model = list(all_model_data.keys())[0]
    model_config = model_results[first_model]
    target_columns = model_config["target_columns"]
    direction_tags = model_config["direction_tags"]
    range_tags = model_config["range_tags"]

    print(f"\n{'-'*20} Generating combined comparison plots {'-'*20}")

    # 1. Plot optimization curves
    if "curves" in plot_types:
        plot_optimization_curves(all_model_data, target_columns, direction_tags, range_tags, output_dir)

    # 2. Plot hypervolume coverage
    if "hv" in plot_types and full_space_file and Path(full_space_file).exists():
        plot_hypervolume_coverage(all_model_data, target_columns, direction_tags, range_tags, Path(full_space_file), output_dir)

    # 3. Plot final best distribution
    if "boxplot" in plot_types:
        plot_final_distribution_boxplot(all_model_data, target_columns, direction_tags, range_tags, output_dir)

    # 4. Plot optimization process scatter(Pareto front comparison)(all models on one figure)
    assert Path(full_space_file).exists(), Path(full_space_file)
    if "scatter" in plot_types and full_space_file and Path(full_space_file).exists():
        plot_optimization_process_scatter(all_model_data, target_columns, direction_tags, range_tags, Path(full_space_file), output_dir)

    print(f"\n{'='*20} All comparison plots completed {'='*20}")
    print(f"Results saved to: {output_dir}")


# Example usage
if __name__ == "__main__":
    # Example: compare synbo, EDBOplus and Gryffin results
    model_results = {
        "synbo": {
            "results_path": "results/multiple_20260320_153018/all_batches_final_round_0.csv",
            "target_columns": ["yield", "cost"],
            "direction_tags": ["max", "min"],
            "range_tags": [[0, 100], [0, 0.1]],
        },
        "EDBOplus": {
            "results_path": "compare_mothods/edboplus/results/batch_1.csv",
            "target_columns": ["yield", "cost"],
            "direction_tags": ["max", "min"],
            "range_tags": [[0, 100], [0, 0.1]],
        },
        "Gryffin": {
            "results_path": "compare_mothods/gryffin/results/merged_Gryffin_for_B-H_HTE.csv",
            "target_columns": ["yield", "cost"],
            "direction_tags": ["max", "min"],
            "range_tags": [[0, 100], [0, 0.1]],
        },
        "LLM (Gemini3-pro)": {
            "results_path": "compare_mothods/LLM/results/final_results-gemini-3-pro.csv",
            "target_columns": ["yield", "cost"],
            "direction_tags": ["max", "min"],
            "range_tags": [[0, 100], [0, 0.1]],
        },
        "LLM (Claude-sonnet-4.6)": {
            "results_path": "compare_mothods/LLM/results/final_results-claude-sonnet-4.6.csv",
            "target_columns": ["yield", "cost"],
            "direction_tags": ["max", "min"],
            "range_tags": [[0, 100], [0, 0.1]],
        },
        "LLM (glm-5)": {
            "results_path": "compare_mothods/LLM/results/final_results-glm5.csv",
            "target_columns": ["yield", "cost"],
            "direction_tags": ["max", "min"],
            "range_tags": [[0, 100], [0, 0.1]],
        },
    }

    full_space_file = "datasets/HTE_datasets/B-H_HTE/B-H_HTE.csv"
    output_dir = "comparison_results"

    plot_comparison(
        model_results=model_results,
        output_dir=output_dir,
        full_space_file=full_space_file,
        plot_types=["curves", "hv", "boxplot", "scatter"],
    )
