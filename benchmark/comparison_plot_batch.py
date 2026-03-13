import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import glob

from utils.plots_batch import (
    plot_optimization_curves,
    plot_hypervolume_coverage,
    plot_final_distribution_boxplot,
    plot_optimization_process_scatter,
)


def plot_comparison(
    model_results: Dict[str, Dict],
    output_dir: str,
    full_space_file: Optional[str] = None,
    plot_types: List[str] = ["curves", "hv", "boxplot", "scatter"],
):
    """
    绘制多个模型的比较结果

    Args:
        model_results: 模型结果的字典，格式为:
            {
                "model_name": {
                    "results_path": "path/to/results.csv",  # 支持通配符，如 "results/multiple_*/all_batches_final_round_*.csv"
                    "target_columns": ["yield", "cost"],    # 需要评估的目标值列名
                    "direction_tags": ["max", "min"],       # 优化方向: "max" 或 "min"
                    "range_tags": [[0, 100], [0, 0.1]],     # 目标值的范围
                },
                ...
            }
        output_dir: 输出目录
        full_space_file: 全空间数据文件路径(用于HV和Pareto前沿计算)
        plot_types: 要绘制的图表类型列表，可选: "curves", "hv", "boxplot", "scatter"

    Returns:
        None (保存图表到output_dir)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*20} Starting Comparison Plotting {'='*20}")
    print(f"Output Directory: {output_dir}")

    # 加载所有模型的数据
    all_model_data = {}
    for model_name, model_config in model_results.items():
        results_path = model_config["results_path"]

        # 使用glob查找匹配的文件
        matching_files = glob.glob(results_path)

        if not matching_files:
            print(f"Warning: No files found for {model_name} with pattern: {results_path}")
            continue

        print(f"Found {len(matching_files)} files for {model_name}")

        # 存储该模型的所有runs
        model_runs = []

        for file_path in matching_files:

            df = pd.read_csv(file_path)

            model_runs.append(df)

        if not model_runs:
            print(f"Warning: No valid data loaded for {model_name}")
            continue

        # 将该模型的所有runs存储到all_model_data
        all_model_data[model_name] = model_runs
        print(f"Loaded {len(model_runs)} runs for {model_name}")

    if not all_model_data:
        print("Error: No valid model data found to plot.")
        return

    print(f"\nLoaded {len(all_model_data)} models for comparison")

    # 设置Seaborn风格
    sns.set_theme(style="whitegrid")

    # 获取配置信息（所有模型应该有相同的配置）
    first_model = list(all_model_data.keys())[0]
    model_config = model_results[first_model]
    target_columns = model_config["target_columns"]
    direction_tags = model_config["direction_tags"]
    range_tags = model_config["range_tags"]

    print(f"\n{'-'*20} Generating combined comparison plots {'-'*20}")

    # 1. 绘制优化曲线（所有模型在同一张图上，带置信区间）
    if "curves" in plot_types:
        plot_optimization_curves(all_model_data, target_columns, direction_tags, range_tags, output_dir)

    # 2. 绘制超体积占比（所有模型在同一张图上）
    if "hv" in plot_types and full_space_file and Path(full_space_file).exists():
        plot_hypervolume_coverage(all_model_data, target_columns, direction_tags, range_tags, Path(full_space_file), output_dir)

    # 3. 绘制最终最佳值分布（所有模型在同一张图上）
    if "boxplot" in plot_types:
        plot_final_distribution_boxplot(all_model_data, target_columns, direction_tags, range_tags, output_dir)

    # 4. 绘制优化过程散点图(Pareto前沿比较)（所有模型在同一张图上）
    if "scatter" in plot_types and full_space_file and Path(full_space_file).exists():
        plot_optimization_process_scatter(all_model_data, target_columns, direction_tags, range_tags, Path(full_space_file), output_dir)

    print(f"\n{'='*20} All comparison plots completed {'='*20}")
    print(f"Results saved to: {output_dir}")


# 示例用法
if __name__ == "__main__":
    # 示例：比较rxnopt和EDBOplus的结果
    # 每个模型的结果从多个CSV文件读取，每个文件代表一个独立的run
    model_results = {
        "rxnopt(dynamic)": {
            "results_path": "results/multiple_20260313_164351/all_batches_final_round_*.csv",
            "target_columns": ["yield", "cost"],
            "direction_tags": ["max", "min"],
            "range_tags": [[0, 100], [0, 0.1]],
        },
        "rxnopt(0.2)": {
            "results_path": "results/multiple_20260309_200135/all_batches_final_round_*.csv",
            "target_columns": ["yield", "cost"],
            "direction_tags": ["max", "min"],
            "range_tags": [[0, 100], [0, 0.1]],
        },
        "rxnopt(0.1)": {
            "results_path": "results/multiple_20260309_194409/all_batches_final_round_*.csv",
            "target_columns": ["yield", "cost"],
            "direction_tags": ["max", "min"],
            "range_tags": [[0, 100], [0, 0.1]],
        },
        "EDBOplus": {
            "results_path": "compare_mothods/edboplus/results/EDBOplus_for_B-H_HTE/batch_*.csv",
            "target_columns": ["yield_collected_values", "cost_collected_values"],
            "direction_tags": ["max", "min"],
            "range_tags": [[0, 100], [0, 0.1]],
        },
    }

    full_space_file = "datasets/HTE_datasets/B-H_HTE/B-H_HTE.csv"
    output_dir = "comparison_results"

    plot_comparison(
        model_results=model_results,
        output_dir=output_dir,
        full_space_file=full_space_file,
        plot_types=["curves", "hv", "boxplot", "scatter"],
    )
