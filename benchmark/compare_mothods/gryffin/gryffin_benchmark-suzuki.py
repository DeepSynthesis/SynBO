"""
Gryffin Benchmark Script with Multiple Random Seeds
This script runs Gryffin benchmarks with multiple random seeds for the suzuki_HTE dataset.

Usage:
    conda activate gryffin_env
    python benchmark/compare_mothods/gryffin/gryffin_benchmark.py
"""

from pathlib import Path
import pandas as pd
import time
from datetime import datetime
import numpy as np
from sklearn.decomposition import PCA
import json
from sklearn.preprocessing import StandardScaler

from gryffin import Gryffin


def build_category_descriptors(desc_dfs, desc_columns, n_components=None, variance_threshold=0.95):
    """
    Build category from descriptor CSV files->descriptor mapping, and reduce dimensions with PCA。

    Parameters:
    -----------
    desc_dfs : dict
        {Column name: DataFrame}, each DataFrame index is category name, columns are descriptors
    desc_columns : list
        List of column names to build descriptors
    n_components : int or None
        Number of dimensions to retain for PCA. None when Automatically determined by variance_threshold。
    variance_threshold : float
        Variance ratio to retain when determining PCA dimensions（default 95%）
    """
    category_descriptors = {}

    for col in desc_columns:
        desc_table = desc_dfs[col].select_dtypes(include=["float64", "float32", "int64"])

        # Delete descriptor columns with zero variance
        desc_table = desc_table.loc[:, desc_table.std() > 0]

        n_cats = len(desc_table)
        n_desc = desc_table.shape[1]

        # Standardization
        scaler = StandardScaler()
        desc_scaled = scaler.fit_transform(desc_table.values)

        # PCA dimensionality reduction：Retain dimensions up to min(num_categories-1, num_descriptors)
        max_components = min(n_cats - 1, n_desc)

        if n_components is not None:
            n_comp = min(n_components, max_components)
        else:
            pca_full = PCA(n_components=max_components)
            pca_full.fit(desc_scaled)
            cumvar = np.cumsum(pca_full.explained_variance_ratio_)
            n_comp = int(np.searchsorted(cumvar, variance_threshold) + 1)
            n_comp = min(n_comp, max_components)

        pca = PCA(n_components=n_comp)
        desc_reduced = pca.fit_transform(desc_scaled)

        explained = pca.explained_variance_ratio_.sum()
        print(
            f"  [{col}] {n_cats} categories: " f"{n_desc} descriptors → {n_comp} PCA components " f"(explained variance: {explained:.3f})"
        )

        category_details = {}
        for i, cat_name in enumerate(desc_table.index):
            category_details[str(cat_name)] = desc_reduced[i].tolist()

        category_descriptors[col] = category_details

    return category_descriptors


def create_gryffin_config(df_ground, category_descriptors, desc_columns, batch_size, random_seed):
    """
    Create Gryffin configuration.

    Key: sampling_strategies = batch_size，makes recommend() return batch_size different candidate points。
    """
    config = {
        "general": {
            "num_cpus": 24,
            "auto_desc_gen": False,
            "batches": 1,
            "sampling_strategies": batch_size,  # ✅ Determines number of candidate points
            "boosted": False,
            "caching": True,
            "random_seed": random_seed,
            "acquisition_optimizer": "genetic",  # 使用遗传算法增强探索
            "dist_param": 0.2,  # 降低密度惩罚，增强探索稀疏区域
            "verbosity": 3,
        },
        "parameters": [],
        "objectives": [
            {"name": "Conversion", "goal": "max", "tolerance": 1.0, "absolute": False},
        ],
    }

    param_columns = [col for col in df_ground.columns if col not in ["new_index", "Conversion"]]

    for col in param_columns:
        if col in desc_columns:
            category_details = category_descriptors[col]

            desc_matrix = np.array(list(category_details.values()))
            if desc_matrix.std() == 0:
                print(f"Warning: {col} descriptors have no variation, skipping...")
                continue

            config["parameters"].append(
                {
                    "name": col,
                    "type": "categorical",
                    "category_details": category_details,
                }
            )
            print(
                f"  [{col}] -> categorical with descriptors "
                f"({len(category_details)} categories, "
                f"{len(list(category_details.values())[0])} dims)"
            )
        else:
            print(col)
            min_val = df_ground[col].min()
            max_val = df_ground[col].max()

            if min_val == max_val:
                print(f"Warning: {col} has no variation, skipping...")
                continue

            config["parameters"].append(
                {
                    "name": col,
                    "type": "continuous",
                    "low": float(min_val),
                    "high": float(max_val),
                }
            )
            print(f"  [{col}] -> continuous [{min_val}, {max_val}]")

    return config


def evaluate_experiment(df_origin, df_ground, params_dict, category_descriptors, desc_columns):
    """
    Exact match for categorical variables，for nearest neighbor match for continuous variables。
    """
    cat_cols = [col for col in params_dict.keys() if col in desc_columns]
    cont_cols = [col for col in params_dict.keys() if col not in desc_columns]

    mask = pd.Series([True] * len(df_origin), index=df_origin.index)
    for col in cat_cols:
        mask &= df_origin[col] == params_dict[col]

    df_filtered_origin = df_origin[mask]
    df_filtered_ground = df_ground[mask]

    if len(df_filtered_origin) == 0:
        print(f"  Warning: No exact match found, falling back to nearest neighbor.")
        df_filtered_origin = df_origin
        df_filtered_ground = df_ground

    if cont_cols:
        cont_vals = np.array([params_dict[c] for c in cont_cols])
        distances = np.sqrt(((df_filtered_ground[cont_cols].values - cont_vals) ** 2).sum(axis=1))
        closest_idx = distances.argmin()
    else:
        closest_idx = 0

    return (
        df_filtered_origin.iloc[closest_idx],
        df_filtered_ground.iloc[closest_idx][["Conversion"]],
    )


def get_starting_point_params(df_origin, start_indices, desc_columns):
    """
    Get the parameter values for the starting point indices from the dataset.

    Parameters:
    -----------
    df_origin : DataFrame
        Original dataset with all columns
    start_indices : list
        List of indices to use as starting points
    desc_columns : list
        List of categorical descriptor columns

    Returns:
    --------
    list of dict
        List of parameter dictionaries for the starting points
    """
    param_columns = [col for col in df_origin.columns if col not in ["new_index", "Conversion"]]

    starting_params = []
    for idx in start_indices:
        row = df_origin.iloc[idx]
        params = {}
        for col in param_columns:
            if col in desc_columns:
                params[col] = str(row[col])
            else:
                params[col] = float(row[col])
        starting_params.append(params)

    return starting_params


def run_gryffin_benchmark(
    df_origin,
    df_ground,
    config,
    category_descriptors,
    desc_columns,
    budget=60,
    batch_size=5,
    seed=1,
    start_indices=None,
):
    """Run a single Gryffin optimization run with optional starting point indices."""
    gryffin = Gryffin(config_dict=config)

    observations = []
    results = []
    num_iterations = int(budget / batch_size)

    print(f"  Total iterations: {num_iterations} " f"(budget={budget}, batch_size={batch_size})")

    for iteration in range(num_iterations):
        print(f"\n  {'='*50}")
        print(f"  Iteration {iteration + 1}/{num_iterations}")
        print(f"  {'='*50}")

        if len(observations) == 0:
            # First iteration: use starting point indices if provided
            if start_indices is not None and len(start_indices) > 0:
                print(f"  [Starting from provided indices]")
                starting_params = get_starting_point_params(df_origin, start_indices, desc_columns)

                # Use provided starting points for batch_size, or pad with random if needed
                params = starting_params[:batch_size]
                while len(params) < batch_size:
                    p = {}
                    for single_p in config["parameters"]:
                        if single_p["type"] == "categorical":
                            cat_names = list(single_p["category_details"].keys())
                            p[single_p["name"]] = np.random.choice(cat_names)
                        else:
                            p[single_p["name"]] = np.random.uniform(single_p["low"], single_p["high"])
                    params.append(p)
            else:
                print("  [Random initialization]")
                params = []
                for _ in range(batch_size):
                    p = {}
                    for single_p in config["parameters"]:
                        if single_p["type"] == "categorical":
                            cat_names = list(single_p["category_details"].keys())
                            p[single_p["name"]] = np.random.choice(cat_names)
                        else:
                            p[single_p["name"]] = np.random.uniform(single_p["low"], single_p["high"])
                    params.append(p)
        else:
            print(f"  [Gryffin recommend]")
            params = gryffin.recommend(observations=observations)

        # Print and evaluate each set of parameters
        for batch_idx, p in enumerate(params):
            print(f"\n  [Batch {batch_idx + 1}/{batch_size}] Proposed conditions:")
            for k, v in p.items():
                print(f"    {k:20s}: {v}")

            result, obs = evaluate_experiment(
                df_origin,
                df_ground,
                p,
                category_descriptors,
                desc_columns,
            )

            print(f"  → Conversion={obs['Conversion']:.4f}")

            obs_dict = dict(p)
            obs_dict["Conversion"] = float(obs["Conversion"])
            observations.append(obs_dict)
            results.append(result)

    return pd.DataFrame(results)


def demo_multiple_configs(
    dataset="HTE_datasets/suzuki_HTE/suzuki_HTE.csv",
    desc_columns=["solvent", "ligand", "reactant2", "reactant1", "base"],
    num_seeds=10,
):
    """Run Gryffin benchmark with multiple random seeds."""
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    dataset_path = Path(__file__).parent / Path(f"../../datasets/{dataset}")
    df_exp = pd.read_csv(dataset_path)
    df_exp.drop(["product", "catalyst", "source", "reaction_id"], axis=1, inplace=True)
    print(df_exp.head())

    budget = 50
    batch_size = 5

    desc_file = dataset_path.parent / Path("descriptors")
    desc_dfs = {col: pd.read_csv(desc_file / Path(f"{col}_RDKit.csv"), index_col=0) for col in desc_columns}

    print("\nBuilding category descriptors (with PCA)...")
    category_descriptors = build_category_descriptors(
        desc_dfs,
        desc_columns,
        n_components=50,  # 使用 50 维 PCA 组件
        variance_threshold=0.95,
    )

    df_ground = df_exp.copy()

    # Load start_point.json
    start_point_path = dataset_path.parent / "start_point.json"
    try:
        with open(start_point_path, "r") as f:
            start_points = json.load(f)
        print(f"\nLoaded start points from: {start_point_path}")
    except FileNotFoundError:
        print(f"\nWarning: {start_point_path} not found, will use random initialization")
        start_points = {}

    print(f"\nGryffin Benchmark with {num_seeds} Random Seeds")
    print(f"Dataset:         {dataset}")
    print(f"Budget:          {budget} experiments")
    print(f"Batch size:      {batch_size}")
    print(f"Number of seeds: {num_seeds}")
    print(f"Desc columns:    {desc_columns}")
    print(f"Start points:    {'Loaded from ' + str(start_point_path) if start_points else 'Random'}")

    label_benchmark = f"Gryffin_for_{dataset_path.name}"
    all_results = []

    for round_id in range(1, num_seeds + 1):
        seed = round_id
        np.random.seed(seed)

        print(f"\n{'='*60}")
        print(f"[Round {round_id}/{num_seeds}]  Seed = {seed}")
        print(f"{'='*60}")

        # Get start indices for this round
        start_key = f"round{round_id}"
        if start_key in start_points:
            start_indices = start_points[start_key]
            print(f"Using start indices from {start_key}: {start_indices}")
        else:
            print(f"Warning: No start indices found for {start_key}, using random initialization")
            start_indices = None

        print("\nCreating Gryffin configuration...")
        config = create_gryffin_config(df_ground, category_descriptors, desc_columns, batch_size=batch_size, random_seed=seed)

        df_round = run_gryffin_benchmark(
            df_exp,
            df_ground,
            config,
            category_descriptors,
            desc_columns,
            budget=budget,
            batch_size=batch_size,
            seed=seed,
            start_indices=start_indices,
        )
        df_round["round_id"] = round_id
        df_round["seed"] = seed
        all_results.append(df_round)

    end_time = time.time()
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_time = end_time - start_time

    results_dir = Path(__file__).parent / Path("results")
    results_dir.mkdir(exist_ok=True)

    if all_results:
        df_merged = pd.concat(all_results, ignore_index=True)
        merged_filename = results_dir / f"merged_{label_benchmark}"
        df_merged.to_csv(merged_filename, index=False)
        print(f"\nMerged results saved to: {merged_filename}")

    timing_filename = results_dir / f"timing_{dataset_path.name.replace('.csv', '')}.txt"
    with open(timing_filename, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Gryffin Benchmark Timing Information\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Dataset:           {dataset}\n")
        f.write(f"Number of seeds:   {num_seeds}\n")
        f.write(f"Budget per seed:   {budget} experiments\n")
        f.write(f"Total experiments: {num_seeds * budget}\n")
        f.write(f"Desc columns:      {desc_columns}\n\n")
        f.write("-" * 80 + "\n")
        f.write("Timing Information:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Start time:        {start_datetime}\n")
        f.write(f"End time:          {end_datetime}\n")
        f.write(f"Total time:        {total_time:.2f} s " f"({total_time/60:.2f} min)\n")
        f.write(f"Avg time/seed:     {total_time/num_seeds:.2f} s\n")

    print(f"\nTiming saved to: {timing_filename}")
    print(f"Total time: {total_time:.2f} s ({total_time/60:.2f} min)\n")


if __name__ == "__main__":
    demo_multiple_configs()
