import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional

# Unified color palette for line elements
LINE_COLORS = [
    "#7153a1",  # Purple (priority for SynBO)
    "#4f75a3",  # Blue
    "#b45475",  # Red
    "#50527a",  # Dark Blue
]

# Global matplotlib rcParams for publication-quality plots
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 14,
    "axes.titlesize": 0,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
})


def load_batch_csv_data(
    results_pattern: str,
    target_column: str = "Conversion",
    direction: str = "max",
    custom_thresholds: Optional[List[float]] = None,
    num_threshold_points: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load data from CSV files and calculate the number of experiments needed
    to reach specific target values.
    """
    matching_files = glob.glob(results_pattern)

    if not matching_files:
        print(f"Warning: No files found with pattern: {results_pattern}")
        return np.array([]), np.array([]), np.array([])

    all_cum_bests = []

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

    if custom_thresholds is not None:
        eval_thresholds = np.sort(np.array(custom_thresholds))
    else:
        global_min = min([cb[0] for cb in all_cum_bests])
        global_max = max([cb[-1] for cb in all_cum_bests])

        if global_min == global_max:
            eval_thresholds = np.array([global_min])
        else:
            eval_thresholds = np.linspace(global_min, global_max, num_threshold_points)

    thresholds_list = []
    means_list = []
    stds_list = []

    for t in eval_thresholds:
        exps_needed = []
        for cb in all_cum_bests:
            if direction == "max":
                idx = np.where(cb >= t)[0]
            else:
                idx = np.where(cb <= t)[0]

            if len(idx) > 0:
                exps_needed.append(idx[0] + 1)

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
    std_scale: float = 0.5,  # <--- 【新增】：在此处控制标准差的缩放倍数
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
    plt.figure(figsize=(10, 7))

    for method_name, (thresholds, means, stds) in loaded_data.items():
        config = methods_config[method_name]

        # 【新增】：对标准差进行缩放
        scaled_stds = stds * std_scale

        plt.errorbar(
            x=thresholds,
            y=means,
            yerr=scaled_stds,  # <--- 【修改】：使用缩放后的标准差作为误差棒
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

    plt.xlabel(f"Target {target_column}")
    plt.ylabel("Number of Experiments Required")

    plt.tick_params(axis="both", which="major")

    plt.grid(True, linestyle="--", alpha=0.6, linewidth=0.8)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    plt.yscale('log')

    plt.legend(loc="best", framealpha=0.9)
    plt.autoscale(enable=True, axis="both", tight=False)

    save_path = output_dir / f"exp_num_comparison_suzuki.png"
    plt.savefig(save_path)
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
            "color": LINE_COLORS[0],
            "marker": "o",
            "zorder": 10,
        },
        "OFAT": {
            "type": "threshold_json",
            "path": "compare_mothods/ofat/results/ofat_expected_results_suzuki.json",
            "color": LINE_COLORS[1],
            "marker": "s",
            "zorder": 9,
        },
        "Random": {
            "type": "threshold_json",
            "path": "compare_mothods/random/results/random_expected_results_suzuki.json",
            "color": LINE_COLORS[2],
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
        std_scale=0.25,  # <--- 【新增】：可以在这里自由控制你想显示的几倍标准差 (如0.5, 0.2等)
    )
