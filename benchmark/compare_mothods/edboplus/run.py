"""
EDBO+ Demo Script
This script demonstrates how to use the EDBO+ package for reaction optimization.

Usage:
    conda activate env_edbo  (or edbo_env)
    python benchmark/compare_mothods/edboplus/run.py

Reference: Based on the benchmark workflow from the original EDBO+ implementation.
"""

from pathlib import Path
import sys
import os
import shutil
import pandas as pd

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


def demo_multiple_configs(dataset="HTE_datasets/B-H_HTE/B-H_HTE.csv", desc_columns=["base", "ligand", "solvent"]):
    """
    Demo: Run multiple benchmarks with different configurations.
    This demonstrates how to compare different acquisition functions and batch sizes.
    """
    # Load the dataset
    dataset_path = Path(__file__).parent / Path(f"../../datasets/{dataset}")
    df_exp = pd.read_csv(dataset_path)

    # Setup parameters
    sort_column = "new_index"

    objectives = ["yield", "cost"]
    objective_modes = ["max", "min"]
    objective_thresholds = [None, None]
    budget = 60
    seed = 1
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
    config = {"batch": batch_size, "acq": "EHVI", "sampling": "cvtsampling", "steps": int(budget / batch_size)}

    # Create results directory
    if not os.path.exists("results"):
        os.makedirs("results")

    print(f"Running benchmarks")

    # Create benchmark filename
    label_benchmark = f"EDBOplus_for_{dataset_path.name}"

    # Clean up previous files
    for filename in [label_benchmark, f"pred_{label_benchmark}", f"results_{label_benchmark}"]:
        if os.path.exists(filename):
            os.remove(filename)

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
    )

    # Move results
    for filename in [label_benchmark, f"pred_{label_benchmark}", f"results_{label_benchmark}"]:
        if os.path.exists(filename):
            shutil.move(filename, f"results/{filename}")

    # Load final hypervolume
    results_path = f"results/results_{label_benchmark}"
    if os.path.exists(results_path):
        df_results = pd.read_csv(results_path)
        final_hv = df_results.iloc[-1]["hypervolume completed (%)"]
        print(f"      Completed! Final hypervolume: {final_hv:.2f}%")


if __name__ == "__main__":
    # HTE_datasets/B-H_HTE/B-H_HTE.csv
    demo_multiple_configs()
