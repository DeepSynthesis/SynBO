"""
Gryffin Benchmark Script with Multiple Random Seeds
This script runs Gryffin benchmarks with multiple random seeds for the B-H_HTE dataset.

Usage:
    conda activate gryffin_env
    python benchmark/compare_mothods/gryffin/gryffin_benchmark.py

Reference: Based on Gryffin optimization workflow and EDBO+ benchmark structure.
"""

from pathlib import Path
import pandas as pd
import time
from datetime import datetime
import numpy as np


from gryffin import Gryffin


def generate_desc_file(df_ground, desc_dfs, desc_columns):
    """Merge descriptor files with ground truth data."""
    res_df = df_ground.copy()
    existing_cols = [col for col in desc_columns if col in res_df.columns]
    for col in existing_cols:
        desc_table = desc_dfs[col]
        desc_table = desc_table.select_dtypes(include=["float64", "float32", "int64"])
        # Reset index to a column for merging
        desc_table = desc_table.reset_index()
        # Rename the index column to match the column name in df_ground
        desc_table = desc_table.rename(columns={"index": col})
        # Rename all descriptor columns with the prefix (including the col column itself temporarily)
        desc_table = desc_table.rename(columns={c: f"{col}_{c}" for c in desc_table.columns})
        # Now rename back the key column to its original name for merging
        desc_table = desc_table.rename(columns={f"{col}_{col}": col})
        # Merge - now all columns should be unique
        res_df = res_df.merge(desc_table, on=col, how="left")
        # Drop the original categorical column after merging
        res_df.drop(col, axis=1, inplace=True)
    return res_df


def create_gryffin_config(df_ground, batch_size):
    """Create Gryffin configuration based on dataset parameters."""
    config = {
        "general": {
            "num_cpus": 24,
            # "auto_desc_gen": True,
            "batches": 1,
            "sampling_strategies": batch_size,
            "boosted": False,
            "caching": True,
            "random_seed": 10,
            "acquisition_optimizer": "adam",
            "verbosity": 3,
        },
        "parameters": [],
        "objectives": [
            {"name": "yield", "goal": "max", "tolerance": 1.0, "absolute": False},
            {"name": "cost", "goal": "min", "tolerance": 0.01, "absolute": False},
        ],
    }

    # Identify parameter columns (excluding objectives and index)
    param_columns = df_ground.columns.tolist()
    param_columns = [col for col in param_columns if col not in ["new_index", "yield", "cost"]]

    # Get unique values for categorical parameters from original dataset
    # For this dataset, we'll treat all parameters as continuous after encoding
    # since they're already encoded as descriptors

    for col in param_columns:
        min_val = df_ground[col].min()
        max_val = df_ground[col].max()

        # Skip columns where min equals max (no variation)
        if min_val == max_val:
            df_ground.drop(col, axis=1, inplace=True)
            continue

        config["parameters"].append({"name": col, "type": "continuous", "low": float(min_val), "high": float(max_val)})

    return config


def evaluate_experiment(df_origin, df_ground, params_dict):
    """Evaluate experiment by finding the closest match in the dataset."""
    # Convert params to DataFrame
    df_params = pd.DataFrame([params_dict])

    # Find the closest row in the dataset based on Euclidean distance
    # Get feature columns (excluding index and objectives)
    feature_cols = [col for col in df_ground.columns if col not in ["new_index", "yield", "cost"]]

    # Calculate distances
    distances = np.sqrt(((df_ground[feature_cols].values - df_params[feature_cols].values) ** 2).sum(axis=1))

    # Find the closest row
    closest_idx = distances.argmin()

    # Return the objectives from the closest row
    return df_origin.iloc[closest_idx], df_ground.iloc[closest_idx][["yield", "cost"]]


def run_gryffin_benchmark(df_origin, df_ground, config, budget=60, batch_size=5, seed=1):
    """Run a single Gryffin optimization run."""
    # Initialize Gryffin
    gryffin = Gryffin(config_dict=config)

    observations = []
    results = []
    print(budget, batch_size, int(budget / batch_size))
    for iteration in range(int(budget / batch_size)):
        # Query Gryffin for new parameters
        if len(observations) == 0:
            # First iteration: sample random point
            params = []
            for _ in range(batch_size):
                p = {}
                for single_p in config["parameters"]:
                    low = single_p["low"]
                    high = single_p["high"]
                    if low == high:
                        # Handle case where low equals high
                        p[single_p["name"]] = low
                    else:
                        # print(low, high)
                        p[single_p["name"]] = np.random.uniform(low, high)
                params.append(p)
        else:
            # Use Gryffin to recommend parameters
            print(111)
            params = gryffin.recommend(observations=observations)

        # Evaluate the proposed parameters
        for p in params:
            result, obs = evaluate_experiment(df_origin, df_ground, p)
            # Create observation dict for Gryffin
            obs_dict = dict(p)
            obs_dict["yield"] = obs["yield"]
            obs_dict["cost"] = obs["cost"]
            observations.append(obs_dict)
            results.append(result)

    return pd.DataFrame(results)


def demo_multiple_configs(dataset="HTE_datasets/B-H_HTE/B-H_HTE.csv", desc_columns=["base", "ligand", "solvent"], num_seeds=1):
    """Run Gryffin benchmark with multiple random seeds."""
    # Start timing
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Load the dataset
    dataset_path = Path(__file__).parent / Path(f"../../datasets/{dataset}")
    df_exp = pd.read_csv(dataset_path)

    # Setup parameters
    sort_column = "new_index"
    budget = 50
    batch_size = 5

    # Loading descriptors
    desc_file = dataset_path.parent / Path("descriptors")
    desc_dfs = {col: pd.read_csv(desc_file / Path(f"{col}_OneHot.csv"), index_col=0) for col in desc_columns}

    # Merge only categorical descriptors with ground truth
    # Concentration and temperature are already continuous, so keep them as is
    df_ground = generate_desc_file(df_exp, desc_dfs, desc_columns)

    # Ensure concentration and temperature are in the dataframe
    if "concentration" not in df_ground.columns:
        df_ground["concentration"] = df_exp["concentration"].values
    if "temperature" not in df_ground.columns:
        df_ground["temperature"] = df_exp["temperature"].values

    print(f"Gryffin Benchmark with {num_seeds} Random Seeds")
    print(f"Dataset: {dataset}")
    print(f"Budget: {budget} experiments")
    print(f"Number of seeds: {num_seeds}")
    print(f"Features: {df_ground.shape[1]-3} (excluding index and objectives)")

    # Create Gryffin configuration
    config = create_gryffin_config(df_ground, batch_size)

    # Create benchmark filename
    label_benchmark = f"Gryffin_for_{dataset_path.name}"

    # List to store all results
    all_results = []

    # Run benchmarks with different seeds
    for round_id in range(1, num_seeds + 1):
        seed = round_id  # Use round_id as seed
        np.random.seed(seed)

        print(f"\n[Round {round_id}/{num_seeds}] Seed = {seed}")

        # Run Gryffin benchmark
        df_round = run_gryffin_benchmark(df_exp, df_ground, config, budget=budget, batch_size=batch_size, seed=seed)
        df_round["round_id"] = round_id
        df_round["seed"] = seed

        all_results.append(df_round)

        # final_hv = df_round.iloc[-1]["hypervolume completed (%)"]
        # print(f"  Final hypervolume: {final_hv:.2f}%")
        # print(f"  Best yield: {df_round.iloc[-1]['yield_best']:.2f}%")
        # print(f"  Best cost: {df_round.iloc[-1]['cost_best']:.4f}")

    # End timing
    end_time = time.time()
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_time = end_time - start_time

    if all_results:
        df_merged = pd.concat(all_results, ignore_index=True)

        # Ensure results directory exists
        results_dir = Path(__file__).parent / Path("results")
        results_dir.mkdir(exist_ok=True)

        # Save merged results
        merged_filename = results_dir / f"merged_{label_benchmark}"
        df_merged.to_csv(merged_filename, index=False)
        print(f"\nMerged results saved to: {merged_filename}")

        # Print summary statistics
        print(f"\nSummary Statistics ({num_seeds} rounds):")
        print(f"{'-'*80}")

        # final_hvs = df_merged.groupby("round_id")["hypervolume completed (%)"].last()
        # print(f"Final Hypervolume (%):")
        # print(f"  Mean: {final_hvs.mean():.2f}%")
        # print(f"  Std:  {final_hvs.std():.2f}%")
        # print(f"  Min:  {final_hvs.min():.2f}%")
        # print(f"  Max:  {final_hvs.max():.2f}%")

        # Calculate mean performance across rounds
        # df_mean = (
        #     df_merged.groupby("step")
        #     .agg({"hypervolume completed (%)": ["mean", "std"], "yield_best": "mean", "cost_best": "mean", "n_experiments": "mean"})
        #     .reset_index()
        # )
        # df_mean.columns = ["step", "hv_mean", "hv_std", "yield_best_mean", "cost_best_mean", "n_experiments_mean"]

        # # Save mean results
        # mean_filename = results_dir / f"mean_{label_benchmark}"
        # df_mean.to_csv(mean_filename, index=False)
        # print(f"Mean results saved to: {mean_filename}")

    # Save timing information to txt file
    timing_filename = results_dir / f"timing_{dataset_path.name.replace('.csv', '')}.txt"
    with open(timing_filename, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Gryffin Benchmark Timing Information\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Number of seeds: {num_seeds}\n")
        f.write(f"Budget per seed: {budget} experiments\n")
        f.write(f"Total experiments: {num_seeds * budget}\n")
        f.write(f"Features: {df_ground.shape[1]-3} (excluding index and objectives)\n\n")
        f.write("-" * 80 + "\n")
        f.write("Timing Information:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Start time: {start_datetime}\n")
        f.write(f"End time: {end_datetime}\n")
        f.write(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)\n")
        f.write(f"Average time per seed: {total_time/num_seeds:.2f} seconds\n")
        f.write(f"\n")

        # if all_results:
        #     f.write("-" * 80 + "\n")
        #     f.write("Performance Summary:\n")
        #     f.write("-" * 80 + "\n")
        #     f.write(f"Final Hypervolume (%):\n")
        #     f.write(f"  Mean: {final_hvs.mean():.2f}%\n")
        #     f.write(f"  Std:  {final_hvs.std():.2f}%\n")
        #     f.write(f"  Min:  {final_hvs.min():.2f}%\n")
        #     f.write(f"  Max:  {final_hvs.max():.2f}%\n")
        #     f.write("\n")

        #     # Performance per round
        #     f.write("-" * 80 + "\n")
        #     f.write("Performance per Round:\n")
        #     f.write("-" * 80 + "\n")
        #     f.write(f"{'Round':<8} {'Seed':<8} {'Final HV (%)':<15} {'Time Est (s)':<15}\n")
        #     f.write("-" * 80 + "\n")
        #     avg_time_per_seed = total_time / num_seeds
        #     for round_id in range(1, num_seeds + 1):
        #         hv = final_hvs[round_id]
        #         f.write(f"{round_id:<8} {round_id:<8} {hv:<15.2f} {avg_time_per_seed:<15.2f}\n")

        # f.write("\n" + "=" * 80 + "\n")

    print(f"\nTiming information saved to: {timing_filename}")

    print(f"\n{'='*80}")
    print("Benchmark completed successfully!")
    print(f"{'='*80}")
    print(f"\nOutput files:")
    print(f"  - Merged results: {merged_filename}")
    # print(f"  - Mean results: {mean_filename}")
    print(f"  - Timing info: {timing_filename}")
    print(f"\nTotal time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # Run benchmark for B-H_HTE dataset
    demo_multiple_configs()
