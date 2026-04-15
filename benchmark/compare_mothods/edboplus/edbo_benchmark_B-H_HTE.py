"""
EDBO+ Benchmark Script with Multiple Random Seeds
This script runs EDBO+ benchmarks with multiple random seeds and merges results.

Usage:
    conda activate env_edbo  (or edbo_env)
    python benchmark/compare_mothods/edboplus/edbo_benchmark.py

Reference: Based on benchmark workflow from original EDBO+ implementation.
"""

from pathlib import Path
import sys
import os
import shutil
import pandas as pd
import time
from datetime import datetime
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from edbo.plus.benchmark.multiobjective_benchmark import Benchmark


def generate_desc_file(df_ground, desc_dfs, desc_columns):
    res_df = df_ground.copy()
    existing_cols = [col for col in desc_columns if col in res_df.columns]
    for col in existing_cols:
        desc_table = desc_dfs[col]
        desc_table = desc_table.select_dtypes(include=["float64", "float32"])
        res_df = res_df.merge(desc_table, left_on=col, right_index=True, how="left")
        res_df.drop([col], axis=1, inplace=True)
    return res_df


def demo_multiple_configs(dataset="HTE_datasets/B-H_HTE/B-H_HTE.csv", desc_columns=["base", "ligand", "solvent"], num_seeds=10):
    # Start timing
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Load dataset
    dataset_path = Path(__file__).parent / Path(f"../../datasets/{dataset}")
    df_exp = pd.read_csv(dataset_path)

    # Setup parameters
    sort_column = "index"

    objectives = ["yield", "cost"]
    objective_modes = ["max", "min"]
    objective_thresholds = [None, None]
    budget = 50
    batch_size = 5

    # loading descs
    desc_file = dataset_path.parent / Path("descriptors")
    desc_dfs = {col: pd.read_csv(desc_file / Path(f"{col}_dft.csv"), index_col=0) for col in desc_columns}

    df_ground = generate_desc_file(df_exp, desc_dfs, desc_columns)

    columns_regression = df_ground.columns.tolist()
    columns_regression.remove(sort_column)
    columns_regression.remove("yield")
    columns_regression.remove("cost")

    # Define configurations to test
    config = {"batch": batch_size, "acq": "EHVI", "sampling": "with_index", "steps": int(budget / batch_size)}

    print(f"EDBO+ Benchmark with {num_seeds} Random Seeds")
    print(f"Dataset: {dataset}")
    print(f"Budget: {budget} experiments ({config['steps']} steps × {config['batch']} batch size)")
    print(f"Acquisition function: {config['acq']}")
    print(f"Sampling method: {config['sampling']}")
    print(f"Number of seeds: {num_seeds}")

    # Create benchmark filename
    label_benchmark = f"EDBOplus_for_{dataset_path.name}"

    # Load start_point.json
    start_point_path = dataset_path.parent / "start_point.json"
    with open(start_point_path, "r") as f:
        start_points = json.load(f)

    # List to store all results
    all_results = []

    # Run benchmarks with different seeds
    for round_id in range(1, num_seeds + 1):
        seed = round_id  # Use round_id as seed

        print(f"\n[Round {round_id}/{num_seeds}] Seed = {seed}")

        # Clean up previous files
        for filename in [label_benchmark, f"pred_{label_benchmark}", f"results_{label_benchmark}"]:
            if os.path.exists(filename):
                os.remove(filename)

        # Initialize benchmark file WITHOUT objective values (only features)
        # This allows the 'with_index' init method to work properly
        df_init = df_ground.copy()
        df_init = df_init.drop(columns=objectives)
        df_init.to_csv(label_benchmark, index=False)
        print(f"Initialized benchmark file with features only (will use 'with_index' initialization)")

        # Get start indices for this round
        start_key = f"round{round_id}"
        if start_key in start_points:
            start_indices = start_points[start_key]
            print(f"Using start indices from {start_key}: {start_indices}")
        else:
            print(f"Warning: No start indices found for {start_key}, using random initialization")
            start_indices = None
            config["sampling"] = "cvtsampling"

        # Initialize and run benchmark
        bench = Benchmark(
            df_ground=df_ground,
            index_column=sort_column,
            objective_names=objectives,
            objective_modes=objective_modes,
            objective_thresholds=objective_thresholds,
            features_regression=columns_regression,
            filename=label_benchmark,
            filename_results=f"results_{label_benchmark}",
            acquisition_function=config["acq"],
        )

        bench.run(
            steps=config["steps"],
            batch=config["batch"],
            seed=seed,
            plot_ground=False,
            plot_predictions=False,
            plot_train=False,
            init_method=config["sampling"],
            init_indices=start_indices,
        )

        # Load and store results
        results_path = f"results_{label_benchmark}"
        if os.path.exists(results_path):
            df_round = pd.read_csv(results_path)
            df_round["round_id"] = round_id
            df_round["seed"] = seed
            all_results.append(df_round)

            final_hv = df_round.iloc[-1]["hypervolume completed (%)"]
            print(f"  Final hypervolume: {final_hv:.2f}%")

            # Remove individual result file to save space
            os.remove(results_path)

        # Clean up other files
        for filename in [label_benchmark, f"pred_{label_benchmark}"]:
            if os.path.exists(filename):
                os.remove(filename)

    # End timing
    end_time = time.time()
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_time = end_time - start_time

    if all_results:
        df_merged = pd.concat(all_results, ignore_index=True)

        # Save merged results
        # df_merged.to_csv(merged_filename, index=False)
        # print(f"Merged results saved to: {merged_filename}")

        # Print summary statistics
        print(f"\nSummary Statistics ({num_seeds} rounds):")
        print(f"{'-'*80}")

        final_hvs = df_merged.groupby("round_id")["hypervolume completed (%)"].last()
        print(f"Final Hypervolume (%):")
        print(f"  Mean: {final_hvs.mean():.2f}%")
        print(f"  Std:  {final_hvs.std():.2f}%")
        print(f"  Min:  {final_hvs.min():.2f}%")
        print(f"  Max:  {final_hvs.max():.2f}%")

        # Calculate mean performance across rounds
        df_mean = (
            df_merged.groupby("step")
            .agg({"hypervolume completed (%)": ["mean", "std"], "yield_best": "mean", "cost_best": "mean", "n_experiments": "mean"})
            .reset_index()
        )
        df_mean.columns = ["step", "hv_mean", "hv_std", "yield_best_mean", "cost_best_mean", "n_experiments_mean"]

        # Save mean results
        mean_filename = f"results/mean_{label_benchmark}"
        df_mean.to_csv(mean_filename, index=False)
        print(f"Mean results saved to: {mean_filename}")

    # Save timing information to txt file
    timing_filename = f"results/timing_{dataset_path.name.replace('.csv', '')}.txt"
    with open(timing_filename, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("EDBO+ Benchmark Timing Information\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Number of seeds: {num_seeds}\n")
        f.write(f"Budget per seed: {budget} experiments\n")
        f.write(f"Total experiments: {num_seeds * budget}\n")
        f.write(f"Acquisition function: {config['acq']}\n")
        f.write(f"Sampling method: {config['sampling']}\n")
        f.write(f"Start points loaded from: {start_point_path}\n\n")
        f.write("-" * 80 + "\n")
        f.write("Timing Information:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Start time: {start_datetime}\n")
        f.write(f"End time: {end_datetime}\n")
        f.write(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)\n")
        f.write(f"Average time per seed: {total_time/num_seeds:.2f} seconds\n")
        f.write(f"\n")

        if all_results:
            f.write("-" * 80 + "\n")
            f.write("Performance Summary:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Final Hypervolume (%):\n")
            f.write(f"  Mean: {final_hvs.mean():.2f}%\n")
            f.write(f"  Std:  {final_hvs.std():.2f}%\n")
            f.write(f"  Min:  {final_hvs.min():.2f}%\n")
            f.write(f"  Max:  {final_hvs.max():.2f}%\n")
            f.write("\n")

            # Performance per round
            f.write("-" * 80 + "\n")
            f.write("Performance per Round:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Round':<8} {'Seed':<8} {'Final HV (%)':<15} {'Time Est (s)':<15}\n")
            f.write("-" * 80 + "\n")
            avg_time_per_step = total_time / (num_seeds * config["steps"])
            for round_id in range(1, num_seeds + 1):
                hv = final_hvs[round_id]
                f.write(f"{round_id:<8} {round_id:<8} {hv:<15.2f} {avg_time_per_step*config['steps']:<15.2f}\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"\nTiming information saved to: {timing_filename}")

    print(f"\n{'='*80}")
    print("Benchmark completed successfully!")
    print(f"{'='*80}")
    print(f"\nOutput files:")
    # print(f"  - Merged results: {merged_filename}")
    print(f"  - Mean results: {mean_filename}")
    print(f"  - Timing info: {timing_filename}")
    print(f"\nTotal time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # HTE_datasets/B-H_HTE/B-H_HTE.csv
    demo_multiple_configs()
