import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any


def load_batch_csv_data(
    results_pattern: str,
    target_column: str = "Conversion",
    direction: str = "max",
    experiments_per_batch: int = 5,  # 新增：从配置中读取每批次实验数
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load data from CSV files (SynBO style) and compute cumulative best values.
    Correctly aligns by actual batch index and forward-fills missing batches.
    """
    matching_files = glob.glob(results_pattern)

    if not matching_files:
        print(f"Warning: No files found with pattern: {results_pattern}")
        return np.array([]), np.array([])

    all_series = []  # 存放每个 run 的 Series (index=batch, value=cum_best)
    valid_files = 0

    for file_path in matching_files:
        df = pd.read_csv(file_path)
        if target_column not in df.columns:
            continue

        valid_files += 1

        # 核心：按 batch 分组求单批次最优
        if direction == "max":
            batch_best = df.groupby("batch")[target_column].max()
            cumulative_best = batch_best.cummax()
        elif direction == "min":
            batch_best = df.groupby("batch")[target_column].min()
            cumulative_best = batch_best.cummin()
        else:
            raise ValueError(f"Unknown direction '{direction}'")

        # 此时 cumulative_best 是一个 Pandas Series，Index 就是真实的 batch 编号
        all_series.append(cumulative_best)

    if not all_series:
        return np.array([]), np.array([])

    # 1. 对齐：使用 pandas.concat 沿列合并，它会自动根据 Index (真实的 batch 号) 对齐！
    # 如果某个 run 缺少某个 batch，对应位置会变成 NaN
    df_all_runs = pd.concat(all_series, axis=1)

    # 2. 填充：排序 batch 索引，并使用 ffill() 向前填充 NaN
    # (如果 Run A 停在 batch 8，它在 batch 9,10 的值会保持 batch 8 的最大值)
    df_all_runs = df_all_runs.sort_index().ffill()

    # 如果开头有 NaN (比如某个 run 没有 batch 0)，用该 run 之后第一个有效值反向填充
    df_all_runs = df_all_runs.bfill()

    # 3. 聚合：计算每一行 (每一个真实 batch) 跨所有 runs 的平均值
    mean_best_values = df_all_runs.mean(axis=1).values

    # 4. 映射 X 轴：获取真实的 batch 编号序列
    actual_batches = df_all_runs.index.values

    # 按照你的逻辑：第 k 个 batch 对应第 k * experiments_per_batch 个实验
    # 注意：如果你的 batch 是从 0 开始算作第一批次，需要 (actual_batches + 1) * 5
    # 如果你的 batch 在 CSV 里本来就是从 1 开始的，直接 actual_batches * 5 即可
    # 这里我做个智能判断：如果最小的 batch 是 0，自动加 1。
    if actual_batches.min() == 0:
        experiment_numbers = (actual_batches + 1) * experiments_per_batch
    else:
        experiment_numbers = actual_batches * experiments_per_batch

    return experiment_numbers, mean_best_values


def load_threshold_json_data(json_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load baseline method data from JSON file (OFAT/Random style).
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Warning: File not found: {json_path}")
        return np.array([]), np.array([])

    thresholds = []
    mean_experiments = []

    for key, value in data.items():
        thresholds.append(float(value["threshold"]))
        mean_experiments.append(float(value["mean"]))

    sorted_indices = np.argsort(thresholds)
    thresholds = np.array(thresholds)[sorted_indices]
    mean_experiments = np.array(mean_experiments)[sorted_indices]

    return mean_experiments, thresholds


def plot_experiment_comparison(
    methods_config: Dict[str, Dict[str, Any]],
    output_dir: str = "comparison_results/exp_num_comparison",
    target_column: str = "Conversion",
    direction: str = "max",
    reference_method: str = "SynBO",
):
    """
    Generate comparison plot of experiment efficiency dynamically based on config.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*20} Starting Experiment Number Comparison {'='*20}")

    # 1. Load Data Dynamically
    loaded_data = {}
    for method_name, config in methods_config.items():
        print(f"Loading data for {method_name}...")
        data_type = config.get("type")
        path = config.get("path")

        if data_type == "batch_csv":
            exp_nums, values = load_batch_csv_data(path, target_column, direction)
        elif data_type == "threshold_json":
            exp_nums, values = load_threshold_json_data(path)
        else:
            print(f"Warning: Unknown data type '{data_type}' for {method_name}")
            continue

        if len(exp_nums) > 0:
            loaded_data[method_name] = (exp_nums, values)
            print(f"  -> Loaded {len(exp_nums)} data points.")
        else:
            print(f"  -> No data loaded for {method_name}.")

    if not loaded_data:
        print("No valid data loaded. Exiting.")
        return

    # 2. Plotting
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 7))

    for method_name, (exp_nums, values) in loaded_data.items():
        config = methods_config[method_name]
        plt.plot(
            exp_nums,
            values,
            marker=config.get("marker", "o"),
            markersize=6,
            linewidth=2.5,
            label=method_name,
            color=config.get("color"),
            zorder=config.get("zorder", 5),
        )

    plt.xlabel("Number of Experiments", fontsize=14, fontname="Arial", fontweight="bold")
    plt.ylabel(f"Expected {target_column}", fontsize=14, fontname="Arial", fontweight="bold")
    plt.title(f"Experiment Efficiency Comparison: {target_column}", fontsize=16, fontname="Arial", fontweight="bold")

    plt.tick_params(axis="both", which="major", labelsize=12, width=1.5, length=6)
    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        label.set_fontname("Arial")

    plt.grid(True, linestyle="--", alpha=0.6, linewidth=0.8)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    plt.legend(loc="best", fontsize=12, framealpha=0.9)

    save_path = output_dir / f"exp_num_comparison_{target_column.lower()}.png"
    plt.ylim(90, 100)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved: {save_path}")

    # 3. Summary Statistics
    print(f"\n{'='*20} Summary Statistics {'='*20}")
    for method_name, (exp_nums, values) in loaded_data.items():
        print(f"{method_name} final/max: {values[-1]:.2f} at {exp_nums[-1]:.1f} experiments")

    # 4. Efficiency Analysis (Compared to Reference Method)
    print(f"\n{'='*20} Efficiency Analysis {'='*20}")
    if reference_method in loaded_data:
        ref_exp_nums, ref_values = loaded_data[reference_method]
        comparison_thresholds = [90, 92, 94, 95]

        for threshold in comparison_thresholds:
            ref_idx = np.where(ref_values >= threshold)[0]
            if len(ref_idx) == 0:
                continue

            ref_exp_needed = ref_exp_nums[ref_idx[0]]
            print(f"\nTo achieve {target_column} ≥ {threshold}:")
            print(f"  - {reference_method} needs: {ref_exp_needed:.1f} experiments")

            for method_name, (exp_nums, values) in loaded_data.items():
                if method_name == reference_method:
                    continue

                other_idx = np.where(values >= threshold)[0]
                if len(other_idx) > 0:
                    other_exp_needed = exp_nums[other_idx[0]]
                    savings = (other_exp_needed - ref_exp_needed) / other_exp_needed * 100
                    print(f"  - {method_name} needs: {other_exp_needed:.1f} experiments")
                    print(f"  - {reference_method} saves {savings:.1f}% experiments vs {method_name}")
                else:
                    print(f"  - {method_name}: threshold not reached")

    print(f"\n{'='*20} Comparison Complete {'='*20}")


if __name__ == "__main__":
    # ==========================================
    # 核心配置字典 (Configuration Dictionary)
    # 若要新增数据，只需在这里添加一个新的字典项即可
    # type: "batch_csv" (读取多轮实验CSV) 或 "threshold_json" (读取OFAT/Random的JSON)
    # ==========================================
    methods_config = {
        "SynBO": {
            "type": "batch_csv",
            "path": "results/multiple_20260417_133009/all_batches_final_round_*.csv",
            "color": "#2E86AB",
            "marker": "o",
            "zorder": 10,
        },
        # "EDBO": {
        #     "type": "batch_csv",
        #     "path": "compare_mothods/edboplus/results/EDBOplus_for_suzuki_HTE/batch_*.csv",
        #     "color": "#2D9255",
        #     "marker": "o",
        #     "zorder": 10,
        # },
        # "Gryffin": {
        #     "type": "batch_csv",
        #     "path": "compare_mothods/gryffin/results/suzuki_HTE/batch_*.csv",
        #     "color": "#772D92",
        #     "marker": "o",
        #     "zorder": 10,
        # },
        "OFAT": {
            "type": "threshold_json",
            "path": "compare_mothods/ofat/results/ofat_expected_results.json",
            "color": "#A23B72",
            "marker": "s",
            "zorder": 9,
        },
        "Random": {
            "type": "threshold_json",
            "path": "compare_mothods/random/results/random_expected_results.json",
            "color": "#F18F01",
            "marker": "^",
            "zorder": 8,
        },
        # 如果你想加入传统的 BO (Bayesian Optimization)，只需取消下面的注释并修改路径：
        # , "TraditionalBO": {
        #     "type": "batch_csv",
        #     "path": "results_bo/multiple_*/all_batches_*.csv",
        #     "color": "#43AA8B",
        #     "marker": "D",
        #     "zorder": 7
        # }
    }

    # Generate comparison plot dynamically
    plot_experiment_comparison(
        methods_config=methods_config,
        output_dir="comparison_results/exp_num_comparison",
        target_column="Conversion",
        direction="max",
        reference_method="SynBO",  # 用于计算 "相比于XXX节省了多少实验" 的基准方法
    )
