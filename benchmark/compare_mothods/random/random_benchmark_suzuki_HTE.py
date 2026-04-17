"""
Random Baseline Benchmark Script for suzuki_HTE
This script runs random sampling benchmarks with multiple rounds, 
using the same start points as EDBO+ for fair comparison.

Usage:
    conda activate env_edbo  (or edbo_env)
    python benchmark/compare_mothods/random/random_benchmark_suzuki_HTE.py

Reference: Based on benchmark workflow from EDBO+ implementation.
"""

from pathlib import Path
import sys
import os
import pandas as pd
import numpy as np
import time
from datetime import datetime
import json
import copy
import torch
import pareto
from sklearn.preprocessing import MinMaxScaler
from botorch.utils.multi_objective.hypervolume import Hypervolume


def generate_desc_file(df_ground, desc_dfs, desc_columns):
    """Merge descriptor files with the ground truth dataframe."""
    res_df = df_ground.copy()
    existing_cols = [col for col in desc_columns if col in res_df.columns]
    for col in existing_cols:
        desc_table = desc_dfs[col].select_dtypes(include=["float64", "float32"])
        desc_table.columns = [f"{col}_{c}_{i+1}" for i, c in enumerate(desc_table.columns)]
        res_df = res_df.merge(desc_table, left_on=col, right_index=True, how="left")
        res_df.drop([col], axis=1, inplace=True)
    return res_df


def run_random_sampling(
    df_ground, sort_column, objectives, objective_modes, objective_thresholds,
    batch_size, steps, seed, init_indices, label_benchmark
):
    """
    Run random sampling benchmark.
    This function mimics the Benchmark.run() logic but uses random sampling
    instead of Bayesian optimization.
    """
    filename = label_benchmark
    filename_results = "results_" + label_benchmark
    
    # Get objective values
    objective_values_ground = []
    for obj_idx, obj in enumerate(objectives):
        d = [float(i) for i in df_ground[obj]]
        if objective_modes[obj_idx] == "min":
            d = -np.array(d)
        else:
            d = np.array(d)
        objective_values_ground.append(d.flatten())
    objective_values_ground = np.array(objective_values_ground).T
    
    # Get Pareto points for ground truth
    pareto_ground = pareto.eps_sort(tables=objective_values_ground, objectives=np.arange(len(objectives)), maximize_all=True)
    pareto_ground = np.array(pareto_ground)
    
    # Fit scaler
    scaler_ground = MinMaxScaler()
    scaler_ground.fit(objective_values_ground)
    
    # Calculate hypervolume ground truth
    references = copy.deepcopy(objective_thresholds)
    if references is None:
        references = [None] * len(objectives)
    for i in range(len(references)):
        if references[i] is None:
            references[i] = np.min(objective_values_ground, axis=0)[i]
    
    references_scaled = scaler_ground.transform(np.array([references]))[0]
    pareto_points_scaled = scaler_ground.transform(pareto_ground)
    pareto_torch = torch.Tensor(pareto_points_scaled)
    hv = Hypervolume(ref_point=torch.Tensor(references_scaled))
    hypervolume_ground = hv.compute(pareto_Y=pareto_torch)
    
    # Initialize benchmark file with features only
    df_init = df_ground.copy()
    df_init = df_init.drop(columns=objectives)
    df_init.to_csv(filename, index=False)
    
    # Set random seed
    np.random.seed(seed)
    
    # Track which samples have been selected
    all_indices = df_ground[sort_column].values
    selected_indices = set()
    
    # Initialize with start indices if provided
    if init_indices is not None:
        selected_indices.update(init_indices)
        print(f"  Initialized with {len(init_indices)} start points")
    
    # Results storage
    results_data = []
    
    for step in range(steps):
        print(f"  Step {step + 1}/{steps}")
        
        # Get available (not yet selected) indices
        available_mask = ~np.isin(all_indices, list(selected_indices))
        available_indices = all_indices[available_mask]
        
        # Randomly sample batch_size candidates
        if len(available_indices) >= batch_size:
            batch_indices = np.random.choice(available_indices, size=batch_size, replace=False)
        else:
            batch_indices = np.random.choice(all_indices, size=batch_size, replace=False)
        
        # Add to selected
        selected_indices.update(batch_indices)
        
        # Update benchmark file with new observations
        df_current = pd.read_csv(filename)
        for batch_idx in batch_indices:
            argwhere_df = np.argwhere(df_ground[sort_column].values == batch_idx)[0][0]
            argwhere_current = np.argwhere(df_current[sort_column].values == batch_idx)[0][0]
            for obj in objectives:
                df_current.loc[argwhere_current, obj] = df_ground[obj].values[argwhere_df]
        df_current.to_csv(filename, index=False)
        
        # Calculate current best values
        new_df = df_current.copy()
        cumulative_train_y = []
        best_values_found = []
        for i in range(len(objectives)):
            if objective_modes[i] == "min":
                best_value = pd.to_numeric(new_df[objectives[i]], "coerce").min()
            else:
                best_value = pd.to_numeric(new_df[objectives[i]], "coerce").max()
            best_values_found.append(best_value)
            vals = pd.to_numeric(new_df[objectives[i]], "coerce").dropna().values
            if objective_modes[i] == "min":
                cumulative_train_y.append(-vals)
            else:
                cumulative_train_y.append(vals)
        
        cumulative_train_y = np.reshape(cumulative_train_y, (len(cumulative_train_y), -1)).T
        
        # Calculate Pareto front and hypervolume
        pareto_train = pareto.eps_sort(tables=cumulative_train_y, objectives=np.arange(len(objectives)), maximize_all=True)
        pareto_train = np.array(pareto_train)
        pareto_train_scaled = scaler_ground.transform(pareto_train)
        pareto_train_torch = torch.Tensor(pareto_train_scaled)
        hypervolume_train = hv.compute(pareto_Y=pareto_train_torch)
        
        n_experiments = len(cumulative_train_y)
        hv_percent = (hypervolume_train / hypervolume_ground) * 100
        print(f"    Experiments: {n_experiments}, Hypervolume: {hv_percent:.2f}%")
        
        # Store results for each sample in batch
        for bt, batch_idx in enumerate(batch_indices):
            sample_vals = {}
            for obj in objectives:
                argwhere = np.argwhere(df_ground[sort_column].values == batch_idx)[0][0]
                sample_vals[obj] = df_ground[obj].values[argwhere]
            
            dict_i = {
                "step": step, "init_method": "random_with_start_points", "init_sample": seed,
                "batch": batch_size, "n_experiments": n_experiments,
                "hypervolume_ground": float(hypervolume_ground), "hypervolume_sampled": float(hypervolume_train),
                "hypervolume completed (%)": float(hv_percent), "sample_index": int(batch_idx),
            }
            dict_i.update(sample_vals)
            for i, obj in enumerate(objectives):
                dict_i[obj + "_best"] = float(best_values_found[i])
            results_data.append(dict_i)
    
    # Save results
    if results_data:
        df_results = pd.DataFrame(results_data)
        df_results.to_csv(filename_results, index=False)
        print(f"  Results saved to {filename_results}")
    
    return df_results


def demo_random_benchmark(dataset="HTE_datasets/suzuki_HTE/suzuki_HTE.csv", desc_columns=["solvent", "ligand", "reactant2", "reactant1", "base"], num_seeds=10):
    """Run random sampling benchmark with multiple seeds."""
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dataset_path = Path(__file__).parent / Path(f"../../datasets/{dataset}")
    df_exp = pd.read_csv(dataset_path)
    sort_column = "index"
    objectives = ["Conversion"]
    objective_modes = ["max"]
    objective_thresholds = [None]
    budget = 50
    batch_size = 5
    desc_file = dataset_path.parent / Path("descriptors")
    desc_dfs = {col: pd.read_csv(desc_file / Path(f"{col}_RDKit.csv"), index_col=0) for col in desc_columns}
    df_ground = generate_desc_file(df_exp, desc_dfs, desc_columns)
    columns_regression = df_ground.columns.tolist()
    columns_regression.remove(sort_column)
    columns_regression.remove("Conversion")
    columns_regression.remove("product")
    columns_regression.remove("catalyst")
    columns_regression.remove("source")
    columns_regression.remove("reaction_id")
    config = {"batch": batch_size, "steps": int(budget / batch_size)}

    print(f"Random Baseline Benchmark with {num_seeds} Random Seeds")
    print(f"Dataset: {dataset}")
    print(f"Budget: {budget} experiments ({config['steps']} steps x {config['batch']} batch size)")
    print(f"Sampling method: Random")
    print(f"Number of seeds: {num_seeds}")

    label_benchmark = "Random_for_" + dataset_path.name
    start_point_path = dataset_path.parent / "start_point.json"
    with open(start_point_path, "r") as f:
        start_points = json.load(f)

    all_results = []

    for round_id in range(1, num_seeds + 1):
        seed = round_id
        print(f"\n[Round {round_id}/{num_seeds}] Seed = {seed}")

        for fn in [label_benchmark, "pred_" + label_benchmark, "results_" + label_benchmark]:
            if os.path.exists(fn):
                os.remove(fn)

        start_key = "round" + str(round_id)
        if start_key in start_points:
            start_indices = start_points[start_key]
            print(f"Using start indices from {start_key}: {start_indices}")
        else:
            print(f"Warning: No start indices found for {start_key}")
            start_indices = None

        df_round = run_random_sampling(
            df_ground=df_ground, sort_column=sort_column, objectives=objectives,
            objective_modes=objective_modes, objective_thresholds=objective_thresholds,
            batch_size=config["batch"], steps=config["steps"], seed=seed,
            init_indices=start_indices, label_benchmark=label_benchmark,
        )

        results_path = "results_" + label_benchmark
        if os.path.exists(results_path):
            df_round = pd.read_csv(results_path)
            df_round["round_id"] = round_id
            df_round["seed"] = seed
            all_results.append(df_round)
            final_hv = df_round.iloc[-1]["hypervolume completed (%)"]
            print(f"  Final hypervolume: {final_hv:.2f}%")
            os.remove(results_path)

        for fn in [label_benchmark, "pred_" + label_benchmark]:
            if os.path.exists(fn):
                os.remove(fn)

    end_time = time.time()
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_time = end_time - start_time

    if all_results:
        df_merged = pd.concat(all_results, ignore_index=True)
        print(f"\nSummary Statistics ({num_seeds} rounds):")
        print("-" * 80)
        final_hvs = df_merged.groupby("round_id")["hypervolume completed (%)"].last()
        print(f"Final Hypervolume (%): Mean: {final_hvs.mean():.2f}%, Std: {final_hvs.std():.2f}%, Min: {final_hvs.min():.2f}%, Max: {final_hvs.max():.2f}%")

        df_mean = df_merged.groupby("step").agg({
            "hypervolume completed (%)": ["mean", "std"], "Conversion_best": "mean", "n_experiments": "mean"
        }).reset_index()
        df_mean.columns = ["step", "hv_mean", "hv_std", "Conversion_best_mean", "n_experiments_mean"]
        os.makedirs("results", exist_ok=True)
        mean_filename = "results/mean_" + label_benchmark
        df_mean.to_csv(mean_filename, index=False)
        print(f"Mean results saved to: {mean_filename}")

    os.makedirs("results", exist_ok=True)
    timing_filename = "results/timing_" + dataset_path.name.replace(".csv", "") + ".txt"
    with open(timing_filename, "w") as f:
        f.write("=" * 80 + "\nRandom Baseline Benchmark Timing Information\n" + "=" * 80 + "\n\n")
        f.write(f"Dataset: {dataset}\nNumber of seeds: {num_seeds}\n")
        f.write(f"Budget per seed: {budget} experiments, Total: {num_seeds * budget}\n")
        f.write(f"Sampling: Random, Start points from: {start_point_path}\n\n")
        f.write("-" * 80 + "\nTiming Information:\n" + "-" * 80 + "\n")
        f.write(f"Start: {start_datetime}, End: {end_datetime}\n")
        f.write(f"Total: {total_time:.2f}s ({total_time/60:.2f}min), Avg per seed: {total_time/num_seeds:.2f}s\n\n")
        if all_results:
            f.write("-" * 80 + "\nPerformance Summary:\n" + "-" * 80 + "\n")
            f.write(f"Final Hypervolume (%): Mean={final_hvs.mean():.2f}%, Std={final_hvs.std():.2f}%\n")
            f.write("-" * 80 + "\nPer Round:\n" + "-" * 80 + "\n")
            f.write(f"{'Round':<8} {'Seed':<8} {'Final HV (%)':<15}\n")
            for round_id in range(1, num_seeds + 1):
                hv = final_hvs[round_id]
                f.write(f"{round_id:<8} {round_id:<8} {hv:<15.2f}\n")
        f.write("\n" + "=" * 80 + "\n")

    print(f"\n{'='*80}")
    print("Benchmark completed successfully!")
    print(f"{'='*80}\nOutput: {mean_filename}, {timing_filename}")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f}min)")


if __name__ == "__main__":
    demo_random_benchmark(
        dataset="HTE_datasets/suzuki_HTE/suzuki_HTE.csv",
        desc_columns=["solvent", "ligand", "reactant2", "reactant1", "base"],
        num_seeds=10
    )
