import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional


def load_batch_csv_data(
    results_pattern: str,
    target_column: str = "Conversion",
    direction: str = "max",
    custom_thresholds: Optional[List[float]] = None,  # 新增：允许传入自定义阈值
    num_threshold_points: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load data from CSV files and calculate the number of experiments needed
    to reach specific target values.

    Returns:
        thresholds (X-axis), means (Y-axis), stds (Y-error)
    """
    matching_files = glob.glob(results_pattern)

    if not matching_files:
        print(f"Warning: No files found with pattern: {results_pattern}")
        return np.array([]), np.array([]), np.array([])

    all_cum_bests = []

    # 1. 提取每个 run 的累积最优值序列
    for file_path in matching_files:
        df = pd.read_csv(file_path)
        if target_column not in df.columns:
            continue

        values = df[target_column].values

        if direction == "max":
            cum_best = np.maximum.accumulate(values)
        elif direction == "min":
            cum_best = np.minimum.accumulate(values)
        else:
            raise ValueError(f"Unknown direction '{direction}'")

        all_cum_bests.append(cum_best)

    if not all_cum_bests:
        return np.array([]), np.array([]), np.array([])

    # 2. 获取评估阈值 (Thresholds)
    if custom_thresholds is not None:
        # 如果提供了自定义阈值，则直接使用并排序
        eval_thresholds = np.sort(np.array(custom_thresholds))
    else:
        # 否则动态生成评估阈值
        global_min = min([cb[0] for cb in all_cum_bests])
        global_max = max([cb[-1] for cb in all_cum_bests])

        if global_min == global_max:
            eval_thresholds = np.array([global_min])
        else:
            eval_thresholds = np.linspace(global_min, global_max, num_threshold_points)

    thresholds_list = []
    means_list = []
    stds_list = []

    # 3. 对每一个阈值，计算达到该阈值所需的实验次数
    for t in eval_thresholds:
        exps_needed = []
        for cb in all_cum_bests:
            # 寻找首次满足阈值的索引
            if direction == "max":
                idx = np.where(cb >= t)[0]
            else:
                idx = np.where(cb <= t)[0]

            if len(idx) > 0:
                # 索引从0开始，实验数从1开始
                exps_needed.append(idx[0] + 1)

        # 只有当至少有一个 run 达到了该阈值时，才进行统计
        # 如果自定义的阈值太高(超过了所有实验的最大值)，这里会自动过滤掉，不会报错
        if len(exps_needed) > 0:
            thresholds_list.append(t)
            means_list.append(np.mean(exps_needed))
            stds_list.append(np.std(exps_needed))

    return np.array(thresholds_list), np.array(means_list), np.array(stds_list)


def load_threshold_json_data(json_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load baseline method data from JSON file (OFAT/Random style).
    Expects structure: {"0.55": {"threshold": 0.55, "mean": 12.48, "std": 7.42...}}
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Warning: File not found: {json_path}")
        return np.array([]), np.array([]), np.array([])

    thresholds = []
    means = []
    stds = []

    for key, value in data.items():
        thresholds.append(float(value["threshold"]))
        means.append(float(value["mean"]))
        stds.append(float(value.get("std", 0.0)))

    sorted_indices = np.argsort(thresholds)
    thresholds = np.array(thresholds)[sorted_indices]
    means = np.array(means)[sorted_indices]
    stds = np.array(stds)[sorted_indices]

    return thresholds, means, stds


def plot_experiment_comparison(
    methods_config: Dict[str, Dict[str, Any]],
    output_dir: str = "comparison_results/exp_num_comparison",
    target_column: str = "Conversion",
    direction: str = "max",
    reference_method: str = "SynBO",
):
    """
    Generate comparison plot: Target Threshold vs Required Experiments with Error Bars.
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
            # 提取自定义阈值参数（如果有）
            custom_thresholds = config.get("custom_thresholds", None)
            thresholds, means, stds = load_batch_csv_data(path, target_column, direction, custom_thresholds=custom_thresholds)
        elif data_type == "threshold_json":
            thresholds, means, stds = load_threshold_json_data(path)
        else:
            print(f"Warning: Unknown data type '{data_type}' for {method_name}")
            continue

        if len(thresholds) > 0:
            loaded_data[method_name] = (thresholds, means, stds)
            print(f"  -> Loaded {len(thresholds)} threshold points.")
        else:
            print(f"  -> No data loaded for {method_name}.")

    if not loaded_data:
        print("No valid data loaded. Exiting.")
        return

    # 2. Plotting
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 7))

    for method_name, (thresholds, means, stds) in loaded_data.items():
        config = methods_config[method_name]

        plt.errorbar(
            x=thresholds,
            y=means,
            yerr=stds,
            fmt=config.get("marker", "o") + "-",
            color=config.get("color"),
            label=method_name,
            markersize=6,
            linewidth=2,
            capsize=4,
            capthick=1.5,
            elinewidth=1.5,
            zorder=config.get("zorder", 5),
            alpha=0.85,
        )

    plt.xlabel(f"Target Value ({target_column})", fontsize=14, fontname="Arial", fontweight="bold")
    plt.ylabel("Number of Experiments Required", fontsize=14, fontname="Arial", fontweight="bold")
    plt.title(f"Efficiency to Reach Target: {target_column}", fontsize=16, fontname="Arial", fontweight="bold")

    plt.tick_params(axis="both", which="major", labelsize=12, width=1.5, length=6)
    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        label.set_fontname("Arial")

    plt.grid(True, linestyle="--", alpha=0.6, linewidth=0.8)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    plt.legend(loc="best", fontsize=12, framealpha=0.9)
    plt.autoscale(enable=True, axis="both", tight=False)

    save_path = output_dir / f"threshold_vs_exp_{target_column.lower()}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved: {save_path}")

    # 3. Efficiency Analysis
    print(f"\n{'='*20} Efficiency Analysis {'='*20}")
    if reference_method in loaded_data:
        ref_thresholds, ref_means, _ = loaded_data[reference_method]

        if len(ref_thresholds) > 4:
            eval_indices = np.linspace(0, len(ref_thresholds) - 1, 5, dtype=int)
            comparison_thresholds = ref_thresholds[eval_indices]
        else:
            comparison_thresholds = ref_thresholds

        for target in comparison_thresholds:
            print(f"\nTo achieve {target_column} ≥ {target:.2f}:")

            ref_idx = np.abs(ref_thresholds - target).argmin()
            ref_exp_needed = ref_means[ref_idx]
            print(f"  - {reference_method} needs: ~{ref_exp_needed:.1f} experiments")

            for method_name, (thresholds, means, _) in loaded_data.items():
                if method_name == reference_method:
                    continue

                if len(thresholds) > 0 and thresholds[-1] >= target * 0.99:
                    other_idx = np.abs(thresholds - target).argmin()
                    other_exp_needed = means[other_idx]
                    savings = (other_exp_needed - ref_exp_needed) / other_exp_needed * 100
                    print(f"  - {method_name} needs: ~{other_exp_needed:.1f} experiments")
                    print(f"  - {reference_method} saves {savings:.1f}% experiments vs {method_name}")
                else:
                    print(f"  - {method_name}: threshold not reached")

    print(f"\n{'='*20} Comparison Complete {'='*20}")


if __name__ == "__main__":
    # ==========================================
    # 核心配置字典 (Configuration Dictionary)
    # ==========================================
    methods_config = {
        "SynBO": {
            "type": "batch_csv",
            "path": "results/multiple_20260421_142852/all_batches_final_round_*.csv",
            "custom_thresholds": list(range(80, 90, 2)) + list(range(90, 96, 1)),
            "color": "#2E86AB",
            "marker": "o",
            "zorder": 10,
        },
        "OFAT": {
            "type": "threshold_json",
            "path": "compare_mothods/ofat/results/ofat_expected_results_suzuki.json",
            "color": "#A23B72",
            "marker": "s",
            "zorder": 9,
        },
        "Random": {
            "type": "threshold_json",
            "path": "compare_mothods/random/results/random_expected_results_suzuki.json",
            "color": "#F18F01",
            "marker": "^",
            "zorder": 8,
        },
    }

    # Generate comparison plot dynamically
    plot_experiment_comparison(
        methods_config=methods_config,
        output_dir="comparison_results/exp_num_comparison",
        target_column="Conversion",
        direction="max",
        reference_method="SynBO",
    )
