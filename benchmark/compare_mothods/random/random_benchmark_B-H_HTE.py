"""
Optimized Random Baseline Benchmark Script for B-H_HTE
- Removed all hypervolume/pareto calculations.
- Optimized Memory & I/O: No intermediate CSV writing/reading.
- O(1) DataFrame lookups.
- Outputs BOTH the mean results AND the complete raw optimization results for all seeds.
"""

from pathlib import Path
import os
import pandas as pd
import numpy as np
import time
from datetime import datetime
import json


def generate_desc_file(df_ground, desc_dfs, desc_columns):
    """Merge descriptor files with the ground truth dataframe."""
    res_df = df_ground.copy()
    existing_cols = [col for col in desc_columns if col in res_df.columns]
    for col in existing_cols:
        desc_table = desc_dfs[col]
        desc_table = desc_table.select_dtypes(include=["float64", "float32"])
        res_df = res_df.merge(desc_table, left_on=col, right_index=True, how="left")
        res_df.drop([col], axis=1, inplace=True)
    return res_df


def run_random_sampling(df_ground, sort_column, objectives, objective_modes, batch_size, steps, seed, init_indices):
    """
    Run random sampling benchmark in memory.
    """
    np.random.seed(seed)
    # Build O(1) hash index to avoid slow full table searches in loops
    df_lookup = df_ground.set_index(sort_column)
    all_indices = df_ground[sort_column].values
    # Initialize set of already selected indices
    selected_set = set()
    best_values = [float("-inf") if mode == "max" else float("inf") for mode in objective_modes]

    def update_best(current, new_val, mode):
        if pd.isna(new_val):  # Ignore NaN, consistent with pandas native behavior
            return current
        return max(current, new_val) if mode == "max" else min(current, new_val)

    results_data = []
    n_experiments = 0
    # ================= Fix: correctly record initial point data =================
    if init_indices is not None:
        selected_set.update(init_indices)
        n_experiments = len(selected_set)

        # 1. First compute best value after all initial batch samples (Historical Best)
        for idx in init_indices:
            row = df_lookup.loc[idx]
            for i, obj in enumerate(objectives):
                best_values[i] = update_best(best_values[i], row[obj], objective_modes[i])
        # 2. Record each sample result from the initial batch
        for idx in init_indices:
            row = df_lookup.loc[idx]
            dict_init = {
                "step": 0,  # Initial point recorded as step 0
                "init_method": "random_with_start_points",
                "init_sample": seed,
                "batch": len(init_indices),  # Initial batch size
                "n_experiments": n_experiments,
                "sample_index": int(idx),
            }
            # Record actual value of current sample
            for obj in objectives:
                dict_init[obj] = float(row[obj])
            # Record historical best value at this point
            for i, obj in enumerate(objectives):
                dict_init[f"{obj}_best"] = float(best_values[i])
            results_data.append(dict_init)
    # ==============================================================
    # Determine step index for subsequent loop
    # If initial points exist, loop starts from step 1; otherwise from step 0
    start_step = 1 if init_indices is not None else 0
    end_step = steps if init_indices is not None else steps - 1
    for step in range(start_step, end_step):
        # Find indices that have not been sampled yet
        available_indices = np.array([idx for idx in all_indices if idx not in selected_set])
        # Randomly sample batch_size candidates
        if len(available_indices) >= batch_size:
            batch_indices = np.random.choice(available_indices, size=batch_size, replace=False)
        else:
            batch_indices = np.random.choice(all_indices, size=batch_size, replace=False)
        # 1. Update selected set and experiment count
        for batch_idx in batch_indices:
            selected_set.add(batch_idx)
        n_experiments += len(batch_indices)
        # 2. Compute best value after current batch (Historical Best)
        for batch_idx in batch_indices:
            row = df_lookup.loc[batch_idx]
            for i, obj in enumerate(objectives):
                best_values[i] = update_best(best_values[i], row[obj], objective_modes[i])
        # 3. Record result data for each sample in this batch
        for batch_idx in batch_indices:
            row = df_lookup.loc[batch_idx]
            dict_i = {
                "step": step,
                "init_method": "random_with_start_points",
                "init_sample": seed,
                "batch": batch_size,
                "n_experiments": n_experiments,
                "sample_index": int(batch_idx),
            }
            # Record actual value of current sample
            for obj in objectives:
                dict_i[obj] = float(row[obj])
            # Record historical best value at this point
            for i, obj in enumerate(objectives):
                dict_i[f"{obj}_best"] = float(best_values[i])
            results_data.append(dict_i)
    return pd.DataFrame(results_data)


def demo_random_benchmark(dataset="HTE_datasets/B-H_HTE/B-H_HTE.csv", desc_columns=["base", "ligand", "solvent"], num_seeds=10):
    """Run random sampling benchmark with multiple seeds."""
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    dataset_path = Path(__file__).parent / Path(f"../../datasets/{dataset}")
    df_exp = pd.read_csv(dataset_path)

    sort_column = "index"
    objectives = ["yield", "cost"]
    objective_modes = ["max", "min"]
    budget = 50
    batch_size = 5

    desc_file = dataset_path.parent / Path("descriptors")
    desc_dfs = {col: pd.read_csv(desc_file / Path(f"{col}_dft.csv"), index_col=0) for col in desc_columns}
    df_ground = generate_desc_file(df_exp, desc_dfs, desc_columns)

    config = {"batch": batch_size, "steps": int(budget / batch_size)}

    print(f"Random Baseline Benchmark with {num_seeds} Random Seeds (Optimized)")
    print(f"Dataset: {dataset}")
    print(f"Budget: {budget} experiments ({config['steps']} steps x {config['batch']} batch size)")

    label_benchmark = "Random_for_" + dataset_path.name
    start_point_path = dataset_path.parent / "start_point.json"

    with open(start_point_path, "r") as f:
        start_points = json.load(f)

    all_results = []

    for round_id in range(1, num_seeds + 1):
        seed = round_id
        start_key = "round" + str(round_id)

        if start_key in start_points:
            start_indices = start_points[start_key]
        else:
            print(f"Warning: No start indices found for {start_key}")
            start_indices = None

        print(f"[Round {round_id}/{num_seeds}] Seed = {seed}")

        df_round = run_random_sampling(
            df_ground=df_ground,
            sort_column=sort_column,
            objectives=objectives,
            objective_modes=objective_modes,
            batch_size=config["batch"],
            steps=config["steps"],
            seed=seed,
            init_indices=start_indices,
        )

        df_round["round_id"] = round_id
        df_round["seed"] = seed
        all_results.append(df_round)

        final_yield = df_round.iloc[-1]["yield_best"]
        final_cost = df_round.iloc[-1]["cost_best"]
        print(f"  Final Best Yield: {final_yield:.2f} | Final Best Cost: {final_cost:.2f}")

    end_time = time.time()
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_time = end_time - start_time

    os.makedirs("results", exist_ok=True)

    if all_results:
        # Merge data from all seed runs
        df_merged = pd.concat(all_results, ignore_index=True)

        print(f"\nSummary Statistics ({num_seeds} rounds):")
        print("-" * 80)

        # 1. Export all raw optimization results (including per-seed, per-step sample results and best values)
        all_results_filename = f"results/all_results_{label_benchmark}"
        df_merged.to_csv(all_results_filename, index=False)
        print(f"All raw results (all seeds & steps) saved to: {all_results_filename}")

        # 2. Compute mean and variance per step and export
        agg_dict = {"yield_best": ["mean", "std"], "cost_best": ["mean", "std"], "n_experiments": "mean"}
        df_mean = df_merged.groupby("step").agg(agg_dict).reset_index()
        # Flatten multi-level column names
        df_mean.columns = ["step", "yield_best_mean", "yield_best_std", "cost_best_mean", "cost_best_std", "n_experiments_mean"]

        mean_filename = f"results/mean_{label_benchmark}"
        df_mean.to_csv(mean_filename, index=False)
        print(f"Mean results saved to: {mean_filename}")

        # 3. Aggregate per-round final results and export timing logs
        final_yields = df_merged.groupby("round_id")["yield_best"].last()
        final_costs = df_merged.groupby("round_id")["cost_best"].last()

        timing_filename = f"results/timing_{dataset_path.name.replace('.csv', '')}.txt"
        with open(timing_filename, "w") as f:
            f.write("=" * 80 + "\nRandom Baseline Benchmark Timing Information\n" + "=" * 80 + "\n\n")
            f.write(f"Dataset: {dataset}\nNumber of seeds: {num_seeds}\n")
            f.write(f"Budget per seed: {budget} experiments, Total: {num_seeds * budget}\n")
            f.write(f"Sampling: Random, Start points from: {start_point_path}\n\n")
            f.write("-" * 80 + "\nTiming Information:\n" + "-" * 80 + "\n")
            f.write(f"Start: {start_datetime}, End: {end_datetime}\n")
            f.write(f"Total: {total_time:.4f}s, Avg per seed: {total_time/num_seeds:.4f}s\n\n")

            f.write("-" * 80 + "\nPerformance Summary:\n" + "-" * 80 + "\n")
            f.write(f"Final Yield_best: Mean={final_yields.mean():.2f}, Std={final_yields.std():.2f}\n")
            f.write(f"Final Cost_best : Mean={final_costs.mean():.2f}, Std={final_costs.std():.2f}\n")
            f.write("-" * 80 + "\nPer Round:\n" + "-" * 80 + "\n")
            f.write(f"{'Round':<8} {'Seed':<8} {'Yield_best':<15} {'Cost_best':<15}\n")
            for round_id in range(1, num_seeds + 1):
                f.write(f"{round_id:<8} {round_id:<8} {final_yields[round_id]:<15.2f} {final_costs[round_id]:<15.2f}\n")
            f.write("\n" + "=" * 80 + "\n")

        print(f"\n{'='*80}")
        print("Benchmark completed successfully!")
        print(f"Output: {all_results_filename}, {mean_filename}, {timing_filename}")
        print(f"Total execution time: {total_time:.4f}s")


if __name__ == "__main__":
    demo_random_benchmark()
benchmark()
enchmark()
benchmark()
