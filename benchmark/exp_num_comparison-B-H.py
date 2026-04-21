import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any

# 引入计算 HV 的工具函数
from synbo.utils.hv_calculator import calculate_hypervolume_by_batch


def load_batch_csv_data_hv(
    results_pattern: str,
    opt_metrics: List[str],
    opt_metric_settings: List[Dict],
    experiments_per_batch: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load data from CSV files and compute cumulative Hypervolume (HV) per batch.
    Specifically designed for multi-objective datasets like B-H.
    """
    matching_files = glob.glob(results_pattern)

    if not matching_files:
        print(f"Warning: No files found with pattern: {results_pattern}")
        return np.array([]), np.array([])

    all_series = []  # 存放每个 run 的 Series (index=batch, value=hv)
    valid_files = 0

    for file_path in matching_files:
        df = pd.read_csv(file_path)

        # 确保所需的优化指标列都在 CSV 中
        if not all(m in df.columns for m in opt_metrics):
            print(f"Warning: Missing required metrics in {file_path}")
            continue

        valid_files += 1

        try:
            # 调用 SynBO 的 HV 计算器
            # 注意：这要求 df 中包含 'batch' 列，以区分实验批次
            hv_results = calculate_hypervolume_by_batch(
                prev_rxn_info=df,
                opt_metrics=opt_metrics,
                opt_metric_settings=opt_metric_settings,
                reference_point_multiplier=1.0,
                cummax=True,
            )

            # 提取 batch 和 hv_normalized
            # 如果计算器返回的 DataFrame 包含 'batch' 列，将其设为 Index
            if "batch" in hv_results.columns:
                hv_series = hv_results.drop_duplicates(subset=["batch"], keep="last").set_index("batch")["hv_normalized"]
            else:
                # 如果没有 batch 列，假设每一行对应一个递增的 batch
                hv_series = pd.Series(hv_results["hv_normalized"].values, index=np.arange(len(hv_results)))

            all_series.append(hv_series)

        except Exception as e:
            print(f"Error calculating HV for {file_path}: {e}")
            continue

    if not all_series:
        return np.array([]), np.array([])

    # 1. 对齐：自动根据真实的 batch 编号对齐
    df_all_runs = pd.concat(all_series, axis=1)

    # 2. 填充：排序 batch 索引，向前填充 (ffill) 停滞的 run，并向后填充 (bfill) 缺失的初始 batch
    df_all_runs = df_all_runs.sort_index().ffill().bfill()

    # 3. 聚合：计算每一个真实 batch 跨所有 runs 的平均 HV 值
    mean_hv_values = df_all_runs.mean(axis=1).values

    # 4. 映射 X 轴：计算实验数量
    actual_batches = df_all_runs.index.values
    if actual_batches.min() == 0:
        experiment_numbers = (actual_batches + 1) * experiments_per_batch
    else:
        experiment_numbers = actual_batches * experiments_per_batch

    return experiment_numbers, mean_hv_values


def load_threshold_json_data(json_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load baseline method data from JSON file (OFAT/Random).
    For B-H dataset, the "threshold" in JSON is the HV value.
    Returns: (experiment_numbers, thresholds/HV_values)
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

    # 按 HV 阈值从小到大排序，以便于画图
    sorted_indices = np.argsort(thresholds)
    thresholds = np.array(thresholds)[sorted_indices]
    mean_experiments = np.array(mean_experiments)[sorted_indices]

    return mean_experiments, thresholds


def plot_experiment_comparison_hv(
    methods_config: Dict[str, Dict[str, Any]],
    opt_metrics: List[str],
    opt_metric_settings: List[Dict],
    output_dir: str = "comparison_results/exp_num_comparison",
    reference_method: str = "SynBO",
):
    """
    Generate comparison plot of experiment efficiency specifically for Hypervolume (HV).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*20} Starting Experiment Number Comparison (B-H / HV) {'='*20}")

    # 1. 动态加载数据
    loaded_data = {}
    for method_name, config in methods_config.items():
        print(f"Loading data for {method_name}...")
        data_type = config.get("type")
        path = config.get("path")

        if data_type == "batch_csv":
            exp_nums, values = load_batch_csv_data_hv(
                results_pattern=path,
                opt_metrics=opt_metrics,
                opt_metric_settings=opt_metric_settings,
                experiments_per_batch=config.get("experiments_per_batch", 5),
            )
        elif data_type == "threshold_json":
            exp_nums, values = load_threshold_json_data(path)
        else:
            print(f"Warning: Unknown data type '{data_type}' for {method_name}")
            continue

        if len(exp_nums) > 0:
            loaded_data[method_name] = (exp_nums, values)
            print(f"  -> Loaded {len(exp_nums)} data points (Max HV: {values[-1]:.3f}).")
        else:
            print(f"  -> No data loaded for {method_name}.")

    if not loaded_data:
        print("No valid data loaded. Exiting.")
        return

    # 2. 绘图配置
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

    # 修改 Labels 为 Hypervolume
    plt.xlabel("Number of Experiments", fontsize=14, fontname="Arial", fontweight="bold")
    plt.ylabel("Expected Hypervolume (Normalized)", fontsize=14, fontname="Arial", fontweight="bold")
    plt.title("Experiment Efficiency Comparison: B-H Dataset (Multi-Objective)", fontsize=16, fontname="Arial", fontweight="bold")

    plt.tick_params(axis="both", which="major", labelsize=12, width=1.5, length=6)
    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        label.set_fontname("Arial")

    plt.grid(True, linestyle="--", alpha=0.6, linewidth=0.8)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    plt.legend(loc="lower right", fontsize=12, framealpha=0.9)  # HV 图通常图例在右下角比较好

    # 设置 Y 轴范围为 [0, 1.05]，因为归一化的 HV 最大值为 1
    plt.ylim(0.5, 0.95)

    save_path = output_dir / "exp_num_comparison_bh_hv.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved: {save_path}")

    # 3. 统计摘要
    print(f"\n{'='*20} Summary Statistics {'='*20}")
    for method_name, (exp_nums, values) in loaded_data.items():
        print(f"{method_name} final/max HV: {values[-1]:.3f} at {exp_nums[-1]:.1f} experiments")

    # 4. 效率分析（针对多目标 HV 调整阈值）
    print(f"\n{'='*20} Efficiency Analysis {'='*20}")
    if reference_method in loaded_data:
        ref_exp_nums, ref_values = loaded_data[reference_method]
        # 针对 HV 设定的评估阈值
        comparison_thresholds = [0.6, 0.7, 0.8, 0.9, 0.95]

        for threshold in comparison_thresholds:
            ref_idx = np.where(ref_values >= threshold)[0]
            if len(ref_idx) == 0:
                continue

            ref_exp_needed = ref_exp_nums[ref_idx[0]]
            print(f"\nTo achieve Hypervolume ≥ {threshold}:")
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
    # B-H 数据集特定的多目标配置
    # ==========================================
    BH_OPT_METRICS = ["yield", "cost"]
    BH_OPT_METRIC_SETTINGS = [
        {"opt_direct": "max", "opt_range": [0, 100]},
        {"opt_direct": "min", "opt_range": [0, 1]},
    ]

    # ==========================================
    # 核心方法配置字典
    # ==========================================
    methods_config = {
        "SynBO": {
            "type": "batch_csv",
            "path": "results/multiple_20260417_132000/all_batches_final_round_*.csv",  # 替换为实际路径
            "experiments_per_batch": 5,
            "color": "#2E86AB",
            "marker": "o",
            "zorder": 10,
        },
        "OFAT": {
            "type": "threshold_json",
            "path": "compare_mothods/ofat/results/ofat_expected_results_B-H.json",
            "color": "#A23B72",
            "marker": "s",
            "zorder": 9,
        },
        # "Random": {
        #     "type": "threshold_json",
        #     "path": "compare_mothods/random/results/random_expected_results_B-H.json",
        #     "color": "#F18F01",
        #     "marker": "^",
        #     "zorder": 8,
        # },
    }

    # 执行针对 HV 的对比绘图
    plot_experiment_comparison_hv(
        methods_config=methods_config,
        opt_metrics=BH_OPT_METRICS,
        opt_metric_settings=BH_OPT_METRIC_SETTINGS,
        output_dir="comparison_results/exp_num_comparison",
        reference_method="SynBO",
    )
