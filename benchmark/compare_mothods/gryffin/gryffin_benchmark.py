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


def build_category_descriptors(desc_dfs, desc_columns):
    """
    从描述符CSV文件中构建类别->描述符映射。

    Parameters:
    -----------
    desc_dfs : dict
        {列名: DataFrame}，每个DataFrame的index为类别名，列为描述符
    desc_columns : list
        需要构建描述符的列名列表

    Returns:
    --------
    category_descriptors : dict
        {参数名: {类别名: 描述符列表}}
    """
    category_descriptors = {}

    for col in desc_columns:
        desc_table = desc_dfs[col].select_dtypes(include=["float64", "float32", "int64"])

        # 删除方差为0的描述符列
        desc_table = desc_table.loc[:, desc_table.std() > 0]

        category_details = {}
        for cat_name, row in desc_table.iterrows():
            category_details[str(cat_name)] = row.tolist()

        category_descriptors[col] = category_details
        print(f"  [{col}] {len(category_details)} categories, {desc_table.shape[1]} descriptors each")

    return category_descriptors


def create_gryffin_config(df_ground, category_descriptors, desc_columns, batch_size, random_seed):
    """
    Create Gryffin configuration using categorical parameters with descriptors
    and continuous parameters without descriptors.

    Parameters:
    -----------
    df_ground : pd.DataFrame
        实验数据，包含类别标签列和目标列
    category_descriptors : dict
        {参数名: {类别名: 描述符列表}}
    desc_columns : list
        使用描述符的类别参数列名
    batch_size : int
        每次推荐的参数数量
    random_seed : int
        随机种子
    """
    config = {
        "general": {
            "num_cpus": 24,
            "auto_desc_gen": False,
            "batches": 1,
            "sampling_strategies": batch_size,
            "boosted": False,
            "caching": True,
            "random_seed": random_seed,
            "acquisition_optimizer": "adam",
            "verbosity": 3,
        },
        "parameters": [],
        "objectives": [
            {"name": "yield", "goal": "max", "tolerance": 1.0, "absolute": False},
            {"name": "cost", "goal": "min", "tolerance": 0.01, "absolute": False},
        ],
    }

    param_columns = [col for col in df_ground.columns if col not in ["new_index", "yield", "cost"]]

    for col in param_columns:
        if col in desc_columns:
            # ✅ 类别变量 + 描述符
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
            print(f"  [{col}] -> categorical with descriptors ({len(category_details)} categories)")

        else:
            # ✅ 连续变量（如 concentration, temperature）
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
    Evaluate experiment by finding the closest match in the dataset.
    对类别变量直接精确匹配，对连续变量使用欧式距离。
    """
    cat_cols = [col for col in params_dict.keys() if col in desc_columns]
    cont_cols = [col for col in params_dict.keys() if col not in desc_columns]

    # 先按类别变量精确过滤
    mask = pd.Series([True] * len(df_origin), index=df_origin.index)
    for col in cat_cols:
        mask &= df_origin[col] == params_dict[col]

    df_filtered_origin = df_origin[mask]
    df_filtered_ground = df_ground[mask]

    if len(df_filtered_origin) == 0:
        # 若无精确匹配，回退到全数据集最近邻
        df_filtered_origin = df_origin
        df_filtered_ground = df_ground

    if cont_cols:
        # 在过滤后的数据中，对连续变量计算最近邻
        cont_vals = np.array([params_dict[c] for c in cont_cols])
        distances = np.sqrt(((df_filtered_ground[cont_cols].values - cont_vals) ** 2).sum(axis=1))
        closest_idx = distances.argmin()
    else:
        closest_idx = 0

    return (
        df_filtered_origin.iloc[closest_idx],
        df_filtered_ground.iloc[closest_idx][["yield", "cost"]],
    )


def run_gryffin_benchmark(df_origin, df_ground, config, category_descriptors, desc_columns, budget=60, batch_size=5, seed=1):
    """Run a single Gryffin optimization run."""
    gryffin = Gryffin(config_dict=config)

    observations = []
    results = []

    num_iterations = int(budget / batch_size)
    print(f"  Total iterations: {num_iterations} (budget={budget}, batch_size={batch_size})")

    for iteration in range(num_iterations):
        print(f"\n  --- Iteration {iteration + 1}/{num_iterations} ---")

        if len(observations) == 0:
            # 第一次迭代：随机采样
            params = []
            for _ in range(batch_size):
                p = {}
                for single_p in config["parameters"]:
                    if single_p["type"] == "categorical":
                        cat_names = list(single_p["category_details"].keys())
                        p[single_p["name"]] = np.random.choice(cat_names)
                    else:
                        low = single_p["low"]
                        high = single_p["high"]
                        p[single_p["name"]] = np.random.uniform(low, high)
                params.append(p)
        else:
            # Gryffin 推荐参数
            params = gryffin.recommend(observations=observations)

        # 评估每组参数
        for batch_idx, p in enumerate(params):
            print(f"\n  [Batch {batch_idx + 1}/{batch_size}] Proposed parameters:")
            for k, v in p.items():
                print(f"    {k}: {v}")

            result, obs = evaluate_experiment(df_origin, df_ground, p, category_descriptors, desc_columns)

            print(f"  → Matched experiment:")
            for k, v in result.items():
                print(f"    {k}: {v}")
            print(f"  → yield={obs['yield']:.4f}, cost={obs['cost']:.6f}")

            obs_dict = dict(p)
            obs_dict["yield"] = obs["yield"]
            obs_dict["cost"] = obs["cost"]
            observations.append(obs_dict)
            results.append(result)

    return pd.DataFrame(results)


def demo_multiple_configs(
    dataset="HTE_datasets/B-H_HTE/B-H_HTE.csv",
    desc_columns=["base", "ligand", "solvent"],
    num_seeds=1,
):
    """Run Gryffin benchmark with multiple random seeds."""
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    dataset_path = Path(__file__).parent / Path(f"../../datasets/{dataset}")
    df_exp = pd.read_csv(dataset_path)

    budget = 50
    batch_size = 5

    # 加载描述符文件
    desc_file = dataset_path.parent / Path("descriptors")
    desc_dfs = {col: pd.read_csv(desc_file / Path(f"{col}_dft.csv"), index_col=0) for col in desc_columns}

    # 构建类别->描述符映射
    print("\nBuilding category descriptors...")
    category_descriptors = build_category_descriptors(desc_dfs, desc_columns)

    # df_ground 保留原始类别列（供 Gryffin 使用）+ 连续变量
    df_ground = df_exp.copy()

    # 确保连续变量存在
    for cont_col in ["concentration", "temperature"]:
        if cont_col not in df_ground.columns and cont_col in df_exp.columns:
            df_ground[cont_col] = df_exp[cont_col].values

    print(f"\nGryffin Benchmark with {num_seeds} Random Seeds")
    print(f"Dataset:        {dataset}")
    print(f"Budget:         {budget} experiments")
    print(f"Batch size:     {batch_size}")
    print(f"Number of seeds:{num_seeds}")
    print(f"Desc columns:   {desc_columns}")

    # 创建 Gryffin 配置（使用第一个 seed，后续每轮重新创建）
    print("\nCreating Gryffin configuration...")

    label_benchmark = f"Gryffin_for_{dataset_path.name}"
    all_results = []

    for round_id in range(1, num_seeds + 1):
        seed = round_id
        np.random.seed(seed)

        print(f"\n{'='*60}")
        print(f"[Round {round_id}/{num_seeds}]  Seed = {seed}")
        print(f"{'='*60}")

        # 每轮使用对应 seed 重新创建 config
        config = create_gryffin_config(df_ground, category_descriptors, desc_columns, batch_size, random_seed=seed)

        df_round = run_gryffin_benchmark(
            df_exp,
            df_ground,
            config,
            category_descriptors,
            desc_columns,
            budget=budget,
            batch_size=batch_size,
            seed=seed,
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
        f.write(f"Dataset:              {dataset}\n")
        f.write(f"Number of seeds:      {num_seeds}\n")
        f.write(f"Budget per seed:      {budget} experiments\n")
        f.write(f"Total experiments:    {num_seeds * budget}\n")
        f.write(f"Descriptor columns:   {desc_columns}\n\n")
        f.write("-" * 80 + "\n")
        f.write("Timing Information:\n")
        f.write("-" * 80 + "\n")
        f.write(f"Start time:           {start_datetime}\n")
        f.write(f"End time:             {end_datetime}\n")
        f.write(f"Total time:           {total_time:.2f} seconds ({total_time/60:.2f} minutes)\n")
        f.write(f"Average time/seed:    {total_time/num_seeds:.2f} seconds\n")

    print(f"\nTiming information saved to: {timing_filename}")
    print(f"\n{'='*80}")
    print("Benchmark completed successfully!")
    print(f"{'='*80}")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")


if __name__ == "__main__":
    demo_multiple_configs()
