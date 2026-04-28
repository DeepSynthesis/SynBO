import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional

# 引入计算 HV 的工具函数
from synbo.utils.hv_calculator import calculate_hypervolume_by_batch

# Unified color palette for line elements
LINE_COLORS = [
    "#7153a1",  # Purple (priority for SynBO)
    "#4f75a3",  # Blue
    "#b45475",  # Red
    "#d3991c",  # Dark Blue
]

# Global matplotlib rcParams for publication-quality plots
plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 14,
        "axes.titlesize": 0,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
    }
)


def load_batch_csv_data_hv(
    results_pattern: str,
    opt_metrics: List[str],
    opt_metric_settings: List[Dict],
    experiments_per_batch: int = 5,
    custom_thresholds: Optional[List[float]] = None,
    num_threshold_points: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load data from CSV files and calculate the number of experiments needed
    to reach specific Hypervolume (HV) target values.

    Specifically designed for multi-objective datasets like B-H.
    """
    matching_files = glob.glob(results_pattern)

    if not matching_files:
        print(f"Warning: No files found with pattern: {results_pattern}")
        return np.array([]), np.array([]), np.array([])

    all_cum_hvs = []  # per-run cumulative HV arrays (per-experiment level)

    for file_path in matching_files:
        df = pd.read_csv(file_path)

        # 确保所需的优化指标列都在 CSV 中
        if not all(m in df.columns for m in opt_metrics):
            print(df.columns, opt_metrics)
            print(f"Warning: Missing required metrics in {file_path}")
            continue

        try:
            # 调用 SynBO 的 HV 计算器，获得每个 batch 的累积 HV
            hv_results = calculate_hypervolume_by_batch(
                prev_rxn_info=df,
                opt_metrics=opt_metrics,
                opt_metric_settings=opt_metric_settings,
                reference_point_multiplier=1.0,
                cummax=True,
            )

            # 提取 batch 级别的 hv_normalized
            if "batch" in hv_results.columns:
                hv_df = hv_results.drop_duplicates(subset=["batch"], keep="last").set_index("batch")
                hv_values = hv_df["hv_normalized"].values
            else:
                hv_values = hv_results["hv_normalized"].values

            # 将 batch 级别的 HV 扩展到 experiment 级别
            # 每个 batch 包含 experiments_per_batch 个实验
            cum_hv_per_exp = np.repeat(hv_values, experiments_per_batch)
            all_cum_hvs.append(cum_hv_per_exp)

        except Exception as e:
            print(f"Error calculating HV for {file_path}: {e}")
            continue

    if not all_cum_hvs:
        return np.array([]), np.array([]), np.array([])

    # 确定评估阈值
    if custom_thresholds is not None:
        eval_thresholds = np.sort(np.array(custom_thresholds))
    else:
        global_min = min([ch[0] for ch in all_cum_hvs])
        global_max = max([ch[-1] for ch in all_cum_hvs])

        if global_min == global_max:
            eval_thresholds = np.array([global_min])
        else:
            eval_thresholds = np.linspace(global_min, global_max, num_threshold_points)

    thresholds_list = []
    means_list = []
    stds_list = []

    for t in eval_thresholds:
        exps_needed = []
        for ch in all_cum_hvs:
            # HV: 最大化方向，找到第一个达到阈值的位置
            idx = np.where(ch >= t)[0]
            if len(idx) > 0:
                exps_needed.append(idx[0] + 1)  # +1 转为 1-indexed 实验数

        if len(exps_needed) > 0:
            thresholds_list.append(t)
            means_list.append(np.mean(exps_needed))
            stds_list.append(np.std(exps_needed))

    return np.array(thresholds_list), np.array(means_list), np.array(stds_list)


def load_threshold_json_data(json_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load baseline method data from JSON file (OFAT/Random style).
    Expects structure: {"0.55": {"threshold": 0.55, "mean": 12.48, "std": 7.42...}}
    For B-H dataset, the "threshold" in JSON is the HV value.
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

    # 按 HV 阈值从小到大排序，以便于画图
    sorted_indices = np.argsort(thresholds)
    thresholds = np.array(thresholds)[sorted_indices]
    means = np.array(means)[sorted_indices]
    stds = np.array(stds)[sorted_indices]

    return thresholds, means, stds


def plot_experiment_comparison_hv(
    methods_config: Dict[str, Dict[str, Any]],
    opt_metrics: List[str],
    opt_metric_settings: List[Dict],
    output_dir: str = "comparison_results/exp_num_comparison",
    reference_method: str = "SynBO",
    std_scale: float = 0.5,
):
    """
    Generate comparison plot: Target Hypervolume vs Required Experiments with Error Bars.

    Specifically designed for multi-objective datasets (B-H) using Hypervolume (HV).
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
            custom_thresholds = config.get("custom_thresholds", None)
            thresholds, means, stds = load_batch_csv_data_hv(
                results_pattern=path,
                opt_metrics=opt_metrics,
                opt_metric_settings=opt_metric_settings,
                experiments_per_batch=config.get("experiments_per_batch", 5),
                custom_thresholds=custom_thresholds,
            )
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

    # 2. 绘图
    plt.figure(figsize=(10, 5), constrained_layout=True)

    for method_name, (thresholds, means, stds) in loaded_data.items():
        config = methods_config[method_name]

        # 对标准差进行缩放
        scaled_stds = stds * std_scale

        plt.errorbar(
            x=thresholds,
            y=means,
            yerr=scaled_stds,
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

    plt.xlabel("Target Hypervolume")
    plt.ylabel("Number of Experiments Required")

    plt.tick_params(axis="both", which="major")

    plt.grid(True, linestyle="--", alpha=0.6, linewidth=0.8)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    plt.yscale("log")

    plt.legend(loc="best", framealpha=0.9)
    plt.autoscale(enable=True, axis="both", tight=False)

    save_path = output_dir / "exp_num_comparison_bh_hv.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved: {save_path}")

    # 3. 效率分析
    # print(f"\n{'='*20} Efficiency Analysis {'='*20}")
    # if reference_method in loaded_data:
    #     ref_thresholds, ref_means, _ = loaded_data[reference_method]

    #     if len(ref_thresholds) > 4:
    #         eval_indices = np.linspace(0, len(ref_thresholds) - 1, 5, dtype=int)
    #         comparison_thresholds = ref_thresholds[eval_indices]
    #     else:
    #         comparison_thresholds = ref_thresholds

    #     for target in comparison_thresholds:
    #         print(f"\nTo achieve Hypervolume ≥ {target:.3f}:")

    #         ref_idx = np.abs(ref_thresholds - target).argmin()
    #         ref_exp_needed = ref_means[ref_idx]
    #         print(f"  - {reference_method} needs: ~{ref_exp_needed:.1f} experiments")

    #         for method_name, (thresholds, means, _) in loaded_data.items():
    #             if method_name == reference_method:
    #                 continue

    #             if len(thresholds) > 0 and thresholds[-1] >= target * 0.99:
    #                 other_idx = np.abs(thresholds - target).argmin()
    #                 other_exp_needed = means[other_idx]
    #                 savings = (other_exp_needed - ref_exp_needed) / other_exp_needed * 100
    #                 print(f"  - {method_name} needs: ~{other_exp_needed:.1f} experiments")
    #                 print(f"  - {reference_method} saves {savings:.1f}% experiments vs {method_name}")
    #             else:
    #                 print(f"  - {method_name}: threshold not reached")

    print(f"\n{'='*20} Comparison Complete {'='*20}")


if __name__ == "__main__":
    # ==========================================
    # B-H 数据集特定的参数配置
    # ==========================================
    BH_REAGENT_TYPES = ["concentration", "temperature", "base", "ligand", "solvent"]
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
            "path": "results/multiple_20260421_191745/all_batches_final_round_*.csv",
            "experiments_per_batch": 5,
            "custom_thresholds": [float(a) for a in list(np.arange(0.75, 0.96, 0.01))],
            "color": LINE_COLORS[0],
            "marker": "o",
            "zorder": 10,
        },
        "OFAT": {
            "type": "threshold_json",
            "path": "compare_mothods/ofat/results/ofat_expected_results_B-H.json",
            "color": LINE_COLORS[1],
            "marker": "s",
            "zorder": 9,
        },
        "Random": {
            "type": "threshold_json",
            "path": "compare_mothods/random/results/random_expected_results_B-H.json",
            "color": LINE_COLORS[2],
            "marker": "^",
            "zorder": 8,
        },
    }

    # 执行针对 HV 的对比绘图
    plot_experiment_comparison_hv(
        methods_config=methods_config,
        opt_metrics=BH_OPT_METRICS,
        opt_metric_settings=BH_OPT_METRIC_SETTINGS,
        output_dir="comparison_results/exp_num_comparison",
        reference_method="SynBO",
        std_scale=0.25,
    )
