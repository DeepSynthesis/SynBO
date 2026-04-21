import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, Any


def load_batch_csv_data(
    results_pattern: str,
    target_column: str = "Conversion",
    direction: str = "max",
    experiments_per_batch: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load data from CSV files (SynBO style) and compute cumulative best values and standard deviation.
    Returns: (experiment_numbers, mean_values, std_values)
    """
    matching_files = glob.glob(results_pattern)

    if not matching_files:
        print(f"Warning: No files found with pattern: {results_pattern}")
        return np.array([]), np.array([]), np.array([])

    all_series = []
    valid_files = 0

    for file_path in matching_files:
        df = pd.read_csv(file_path)
        if target_column not in df.columns:
            continue

        valid_files += 1

        if direction == "max":
            batch_best = df.groupby("batch")[target_column].max()
            cumulative_best = batch_best.cummax()
        elif direction == "min":
            batch_best = df.groupby("batch")[target_column].min()
            cumulative_best = batch_best.cummin()
        else:
            raise ValueError(f"Unknown direction '{direction}'")

        all_series.append(cumulative_best)

    if not all_series:
        return np.array([]), np.array([]), np.array([])

    df_all_runs = pd.concat(all_series, axis=1)
    df_all_runs = df_all_runs.sort_index().ffill().bfill()

    # 计算均值和标准差 (跨所有runs)
    mean_best_values = df_all_runs.mean(axis=1).values
    std_best_values = df_all_runs.std(axis=1).fillna(0).values  # 如果只有一个run，std为NaN，填0

    actual_batches = df_all_runs.index.values

    if actual_batches.min() == 0:
        experiment_numbers = (actual_batches + 1) * experiments_per_batch
    else:
        experiment_numbers = actual_batches * experiments_per_batch

    return experiment_numbers, mean_best_values, std_best_values


def load_threshold_json_data(json_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load baseline method data from JSON file (OFAT/Random style) including standard deviation.
    Returns: (mean_experiments, thresholds, std_experiments)
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Warning: File not found: {json_path}")
        return np.array([]), np.array([]), np.array([])

    thresholds = []
    mean_experiments = []
    std_experiments = []

    for key, value in data.items():
        thresholds.append(float(value["threshold"]))
        mean_experiments.append(float(value["mean"]))
        # 读取 std，如果没有则默认为 0
        std_experiments.append(float(value.get("std", 0.0)))

    sorted_indices = np.argsort(thresholds)
    thresholds = np.array(thresholds)[sorted_indices]
    mean_experiments = np.array(mean_experiments)[sorted_indices]
    std_experiments = np.array(std_experiments)[sorted_indices]

    return mean_experiments, thresholds, std_experiments


def plot_experiment_comparison(
    methods_config: Dict[str, Dict[str, Any]],
    output_dir: str = "comparison_results/exp_num_comparison",
    target_column: str = "Conversion",
    direction: str = "max",
    reference_method: str = "SynBO",
):
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
            x, y, y_err = load_batch_csv_data(path, target_column, direction)
            # 对于 CSV 数据，误差在 Y 轴 (目标值波动)
            if len(x) > 0:
                loaded_data[method_name] = {"x": x, "y": y, "x_err": None, "y_err": y_err}
                print(f"  -> Loaded {len(x)} data points.")

        elif data_type == "threshold_json":
            x, y, x_err = load_threshold_json_data(path)
            # 对于 JSON 数据，误差在 X 轴 (需要的实验次数波动)
            if len(x) > 0:
                loaded_data[method_name] = {"x": x, "y": y, "x_err": x_err, "y_err": None}
                print(f"  -> Loaded {len(x)} data points.")
        else:
            print(f"Warning: Unknown data type '{data_type}' for {method_name}")

    if not loaded_data:
        print("No valid data loaded. Exiting.")
        return

    # 2. Plotting
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 7))

    for method_name, data in loaded_data.items():
        config = methods_config[method_name]
        x = data["x"]
        y = data["y"]
        x_err = data["x_err"]
        y_err = data["y_err"]

        base_color = config.get("color")
        zorder = config.get("zorder", 5)

        # 绘制主线
        plt.plot(
            x,
            y,
            marker=config.get("marker", "o"),
            markersize=6,
            linewidth=2.5,
            label=method_name,
            color=base_color,
            zorder=zorder,
        )

        # 绘制误差范围 (Error Bands)
        alpha = 0.2  # 阴影透明度

        if y_err is not None:
            # 垂直误差 (针对 BO 方法)
            plt.fill_between(x, y - y_err, y + y_err, color=base_color, alpha=alpha, zorder=zorder - 1)

        if x_err is not None:
            # 水平误差 (针对 OFAT / Random)
            plt.fill_betweenx(y, x - x_err, x + x_err, color=base_color, alpha=alpha, zorder=zorder - 1)

    plt.xlabel("Number of Experiments", fontsize=14, fontname="Arial", fontweight="bold")
    plt.ylabel(f"Expected {target_column} (with Std. Dev.)", fontsize=14, fontname="Arial", fontweight="bold")
    plt.title(f"Experiment Efficiency Comparison: {target_column}", fontsize=16, fontname="Arial", fontweight="bold")

    plt.tick_params(axis="both", which="major", labelsize=12, width=1.5, length=6)
    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        label.set_fontname("Arial")

    plt.grid(True, linestyle="--", alpha=0.6, linewidth=0.8)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    plt.legend(loc="best", fontsize=12, framealpha=0.9)

    save_path = output_dir / f"exp_num_comparison_{target_column.lower()}_with_errors.png"
    plt.xlim(0, 200)
    plt.ylim(90, 100)  # 根据您的数据视情况修改
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved: {save_path}")

    # 3. Summary Statistics
    print(f"\n{'='*20} Summary Statistics {'='*20}")
    for method_name, data in loaded_data.items():
        x, y = data["x"], data["y"]
        print(f"{method_name} final/max: {y[-1]:.2f} at {x[-1]:.1f} experiments")

    # 4. Efficiency Analysis
    print(f"\n{'='*20} Efficiency Analysis {'='*20}")
    if reference_method in loaded_data:
        ref_data = loaded_data[reference_method]
        ref_x, ref_y = ref_data["x"], ref_data["y"]
        comparison_thresholds = [90, 92, 94, 95]

        for threshold in comparison_thresholds:
            ref_idx = np.where(ref_y >= threshold)[0]
            if len(ref_idx) == 0:
                continue

            ref_exp_needed = ref_x[ref_idx[0]]
            print(f"\nTo achieve {target_column} ≥ {threshold}:")
            print(f"  - {reference_method} needs: {ref_exp_needed:.1f} experiments")

            for method_name, data in loaded_data.items():
                if method_name == reference_method:
                    continue

                x, y = data["x"], data["y"]
                other_idx = np.where(y >= threshold)[0]
                if len(other_idx) > 0:
                    other_exp_needed = x[other_idx[0]]
                    savings = (other_exp_needed - ref_exp_needed) / other_exp_needed * 100
                    print(f"  - {method_name} needs: {other_exp_needed:.1f} experiments")
                    print(f"  - {reference_method} saves {savings:.1f}% experiments vs {method_name}")
                else:
                    print(f"  - {method_name}: threshold not reached")

    print(f"\n{'='*20} Comparison Complete {'='*20}")


if __name__ == "__main__":
    methods_config = {
        "SynBO": {
            "type": "batch_csv",
            "path": "results/multiple_20260417_133009/all_batches_final_round_*.csv",
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

    plot_experiment_comparison(
        methods_config=methods_config,
        output_dir="comparison_results/exp_num_comparison",
        target_column="Conversion",
        direction="max",
        reference_method="SynBO",
    )
