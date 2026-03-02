"""
EDBO+ Demo Script
This script demonstrates how to use the EDBO+ package for reaction optimization.

Usage:
    conda activate env_edbo  (or edbo_env)
    python benchmark/compare_mothods/edboplus/run.py

Reference: Based on the benchmark workflow from the original EDBO+ implementation.
"""

import sys
import os
import shutil
import pandas as pd
import numpy as np

# Add the edbo package to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from edbo.plus.benchmark.multiobjective_benchmark import Benchmark


def demo_single_benchmark():
    """
    Demo: Run a single benchmark with EDBO+ on the B-H_HTE dataset.
    This demonstrates the basic workflow for running a multi-objective optimization.
    """
    print("=" * 80)
    print("DEMO: Single EDBO+ Benchmark Run")
    print("=" * 80)

    # Load the dataset
    dataset_path = "../../datasets/HTE_datasets/B-H_HTE/B-H_HTE.csv"
    df_exp = pd.read_csv(dataset_path)

    print(f"\n1. Loaded dataset: {dataset_path}")
    print(f"   Dataset shape: {df_exp.shape}")
    print(f"   Columns: {df_exp.columns.tolist()}")
    print(f"\n   First 3 rows:")
    print(df_exp.head(3))

    # Define the index column for tracking experiments
    sort_column = "new_index"

    # Define feature columns for regression (all columns except index and objectives)
    columns_regression = df_exp.columns.tolist()
    columns_regression.remove(sort_column)
    columns_regression.remove("yield")
    columns_regression.remove("cost")

    print(f"\n2. Feature columns for regression ({len(columns_regression)}):")
    print(f"   {columns_regression}")

    # Define optimization objectives
    objectives = ["yield", "cost"]  # Two objectives: maximize yield, minimize cost
    objective_modes = ["max", "min"]
    objective_thresholds = [None, None]  # No specific thresholds

    print(f"\n3. Optimization objectives:")
    print(f"   Objectives: {objectives}")
    print(f"   Modes: {objective_modes} (maximize yield, minimize cost)")
    print(f"   Thresholds: {objective_thresholds}")

    # Benchmark parameters
    batch_size = 5
    budget = 60  # Total number of experiments
    steps = int(budget / batch_size)  # Number of iterations
    acquisition_function = "EHVI"  # Expected Hypervolume Improvement
    init_sampling_method = "cvtsampling"  # CVT sampling for initial points
    seed = 1

    # Create benchmark filename
    label_benchmark = f"demo_benchmark_acq_{acquisition_function}_batch_{batch_size}_seed_{seed}_init_{init_sampling_method}.csv"

    print(f"\n4. Benchmark parameters:")
    print(f"   Batch size: {batch_size}")
    print(f"   Total budget: {budget} experiments")
    print(f"   Number of steps: {steps}")
    print(f"   Acquisition function: {acquisition_function}")
    print(f"   Initial sampling: {init_sampling_method}")
    print(f"   Random seed: {seed}")
    print(f"   Results file: {label_benchmark}")

    # Create results directory if it doesn't exist
    if not os.path.exists("results"):
        os.makedirs("results")

    # Clean up any previous files
    print(f"\n5. Cleaning up previous files...")
    for filename in [label_benchmark, f"pred_{label_benchmark}", f"results_{label_benchmark}"]:
        filepath = os.path.join(".", filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"   Removed: {filename}")

    # Initialize and run the benchmark
    print(f"\n6. Initializing EDBO+ Benchmark...")
    bench = Benchmark(
        df_ground=df_exp,
        index_column=sort_column,
        objective_names=objectives,
        objective_modes=objective_modes,
        objective_thresholds=objective_thresholds,
        features_regression=columns_regression,
        filename=label_benchmark,
        filename_results=f"results_{label_benchmark}",
        acquisition_function=acquisition_function,
    )

    print(f"   Ground truth hypervolume: {bench.hypervolume_ground:.6f}")
    print(f"   Number of Pareto optimal points: {len(bench.pareto_ground)}")

    # Run the benchmark
    print(f"\n7. Running optimization (this may take a few minutes)...")
    print(f"   Steps: {steps}, Batch size: {batch_size}")

    bench.run(
        steps=steps,
        batch=batch_size,
        seed=seed,
        plot_ground=False,
        plot_predictions=False,
        plot_train=False,
        init_method=init_sampling_method,
    )

    # Move results to results directory
    print(f"\n8. Saving results to 'results/' directory...")
    for filename in [label_benchmark, f"pred_{label_benchmark}", f"results_{label_benchmark}"]:
        if os.path.exists(filename):
            shutil.move(filename, f"results/{filename}")
            print(f"   Saved: results/{filename}")

    # Load and display results summary
    results_path = f"results/results_{label_benchmark}"
    if os.path.exists(results_path):
        df_results = pd.read_csv(results_path)
        print(f"\n9. Results summary:")
        print(f"   Total experiments: {len(df_results)}")
        print(f"   Steps completed: {df_results['step'].max() + 1}")

        # Show final hypervolume percentage
        final_hv = df_results.iloc[-1]["hypervolume completed (%)"]
        print(f"   Final hypervolume completion: {final_hv:.2f}%")

        # Show best values found
        print(f"\n   Best values found:")
        for obj in objectives:
            best_col = f"{obj}_best"
            if best_col in df_results.columns:
                best_val = df_results[best_col].max()
                print(f"     {obj}: {best_val:.4f}")

        print(f"\n   Last few results:")
        print(df_results[["step", "n_experiments", "hypervolume completed (%)", "yield_best", "cost_best"]].tail(5))

    print(f"\n{'=' * 80}")
    print("Benchmark completed successfully!")
    print(f"Results saved to: results/results_{label_benchmark}")
    print(f"{'=' * 80}\n")


def demo_multiple_configs():
    """
    Demo: Run multiple benchmarks with different configurations.
    This demonstrates how to compare different acquisition functions and batch sizes.
    """
    print("\n" + "=" * 80)
    print("DEMO: Multiple Benchmark Configurations")
    print("=" * 80)

    # Load the dataset
    dataset_path = "../../datasets/HTE_datasets/B-H_HTE/B-H_HTE.csv"
    df_exp = pd.read_csv(dataset_path)

    print(f"\n1. Loaded dataset: {dataset_path}")
    print(f"   Dataset shape: {df_exp.shape}")

    # Setup parameters
    sort_column = "new_index"
    columns_regression = df_exp.columns.tolist()
    columns_regression.remove(sort_column)
    columns_regression.remove("yield")
    columns_regression.remove("cost")

    objectives = ["yield", "cost"]
    objective_modes = ["max", "min"]
    objective_thresholds = [None, None]
    budget = 60
    seed = 1

    # Define configurations to test
    configurations = []

    # Test different batch sizes
    for batch_size in [3, 5, 10]:
        configurations.append({"batch": batch_size, "acq": "EHVI", "sampling": "cvtsampling", "steps": int(budget / batch_size)})

    # Test different sampling methods
    for sampling_method in ["seed", "lhs", "cvtsampling"]:
        configurations.append({"batch": 5, "acq": "EHVI", "sampling": sampling_method, "steps": int(budget / 5)})

    print(f"\n2. Configurations to test ({len(configurations)}):")
    for i, config in enumerate(configurations, 1):
        print(f"   {i}. Batch={config['batch']}, Acq={config['acq']}, " f"Sampling={config['sampling']}, Steps={config['steps']}")

    # Create results directory
    if not os.path.exists("results"):
        os.makedirs("results")

    print(f"\n3. Running benchmarks (this may take several minutes)...")

    results_summary = []

    for i, config in enumerate(configurations, 1):
        print(
            f"\n   [{i}/{len(configurations)}] Running configuration: "
            f"Batch={config['batch']}, Acq={config['acq']}, Sampling={config['sampling']}"
        )

        # Create benchmark filename
        label_benchmark = f"demo_multi_batch_{config['batch']}_" f"acq_{config['acq']}_sampling_{config['sampling']}_" f"seed_{seed}.csv"

        # Clean up previous files
        for filename in [label_benchmark, f"pred_{label_benchmark}", f"results_{label_benchmark}"]:
            if os.path.exists(filename):
                os.remove(filename)

        # Initialize and run benchmark
        bench = Benchmark(
            df_ground=df_exp,
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
            results_summary.append(
                {"batch": config["batch"], "acq": config["acq"], "sampling": config["sampling"], "final_hv_pct": final_hv}
            )
            print(f"      Completed! Final hypervolume: {final_hv:.2f}%")

    # Print summary
    print(f"\n4. Results Summary:")
    print(f"   {'Batch':<6} {'Acq':<6} {'Sampling':<12} {'Hypervolume %':<12}")
    print(f"   {'-'*6} {'-'*6} {'-'*12} {'-'*12}")

    for result in results_summary:
        print(f"   {result['batch']:<6} {result['acq']:<6} {result['sampling']:<12} " f"{result['final_hv_pct']:<12.2f}")

    print(f"\n{'=' * 80}")
    print("Multiple benchmarks completed!")
    print(f"Results saved to: results/")
    print(f"{'=' * 80}\n")


def main():
    """
    Main function to run demos.
    """
    print("\n" + "=" * 80)
    print("EDBO+ DEMO SCRIPT")
    print("=" * 80)
    print("\nThis demo demonstrates how to use the EDBO+ package for")
    print("multi-objective reaction optimization.")

    # Demo 1: Single benchmark run
    print("\n\nRunning Demo 1: Single Benchmark Run")
    print("-" * 80)
    demo_single_benchmark()

    # Uncomment to run Demo 2: Multiple configurations
    # Note: This will take much longer to complete
    print("\n\nRunning Demo 2: Multiple Benchmark Configurations")
    print("-" * 80)
    demo_multiple_configs()

    print("\n" + "=" * 80)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 80)

    print("\nTo run your own benchmarks:")
    print("1. Prepare your dataset with features and objective columns")
    print("2. Define objectives (e.g., ['yield', 'cost'])")
    print("3. Set optimization modes (e.g., ['max', 'min'])")
    print("4. Initialize Benchmark class with your parameters")
    print("5. Call bench.run() to start optimization")
    print("6. Analyze results from the generated CSV files")

    print("\nFor more information, refer to:")
    print("  - edbo/plus/optimizer_botorch.py (main optimizer)")
    print("  - edbo/plus/benchmark/multiobjective_benchmark.py (benchmark)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
