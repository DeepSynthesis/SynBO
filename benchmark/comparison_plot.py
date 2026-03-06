import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional

from utils.plots import (
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
                    "results_path": "path/to/results.csv",  # results.csv文件路径
                    "target_columns": ["yield", "cost"],    # 需要评估的目标值列名
                    "direction_tags": ["max", "min"],       # 优化方向: "max" 或 "min"
                    "range_tags": [[0, 100], [0, 0.5]],     # 目标值的范围
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
        results_path = Path(model_config["results_path"])
        if not results_path.exists():
            print(f"Warning: Results file not found for {model_name}: {results_path}")
            continue

        # 读取结果文件
        df = pd.read_csv(results_path)

        # 标准化列名：将不同格式的列名统一为yield和cost
        # EDBOplus使用yield_collected_values和cost_collected_values
        if "yield_collected_values" in df.columns and "yield" not in df.columns:
            df = df.rename(columns={"yield_collected_values": "yield"})
        if "cost_collected_values" in df.columns and "cost" not in df.columns:
            df = df.rename(columns={"cost_collected_values": "cost"})

        # 更新target_columns为标准化后的列名
        model_config["target_columns"] = [
            "yield" if col == "yield_collected_values" else "cost" if col == "cost_collected_values" else col
            for col in model_config["target_columns"]
        ]

        # rxnopt格式: 包含batch_index和round_index列
        # 可能需要从多个round文件中加载
        if "round_index" in df.columns:
            # 已经是合并后的数据
            all_model_data[model_name] = [df]
        else:
            # 单个round的数据
            df["round_index"] = 0
            all_model_data[model_name] = [df]

    if not all_model_data:
        print("Error: No valid model data found to plot.")
        return

    print(f"Loaded {len(all_model_data)} models for comparison")

    # 设置Seaborn风格
    sns.set_theme(style="whitegrid")

    # 为每个模型生成对比图
    for model_name, dfs in all_model_data.items():
        model_config = model_results[model_name]
        target_columns = model_config["target_columns"]
        direction_tags = model_config["direction_tags"]
        range_tags = model_config["range_tags"]

        model_output_dir = output_dir / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'-'*20} Plotting for {model_name} {'-'*20}")

        # 1. 绘制优化曲线
        if "curves" in plot_types:
            plot_optimization_curves(dfs, target_columns, direction_tags, range_tags, model_output_dir)

        # 2. 绘制超体积占比
        if "hv" in plot_types and full_space_file and Path(full_space_file).exists():
            plot_hypervolume_coverage(dfs, target_columns, direction_tags, range_tags, Path(full_space_file), model_output_dir)

        # 3. 绘制最终最佳值分布
        if "boxplot" in plot_types:
            plot_final_distribution_boxplot(dfs, target_columns, direction_tags, range_tags, model_output_dir)

        # 4. 绘制优化过程散点图(Pareto前沿比较)
        assert Path(full_space_file).exists(), Path(full_space_file)
        if "scatter" in plot_types and full_space_file and Path(full_space_file).exists():
            plot_optimization_process_scatter(dfs, target_columns, direction_tags, range_tags, Path(full_space_file), model_output_dir)

    print(f"\n{'='*20} All comparison plots completed {'='*20}")
    print(f"Results saved to: {output_dir}")


# 示例用法
if __name__ == "__main__":
    # 示例：比较rxnopt, EDBOplus和Gryffin的结果
    model_results = {
        "rxnopt": {
            "results_path": "results/single_20260306_162144/all_batches_final_round_0.csv",
            "target_columns": ["yield", "cost"],
            "direction_tags": ["max", "min"],
            "range_tags": [[0, 100], [0, 0.5]],
        },
        "EDBOplus": {
            "results_path": "compare_mothods/edboplus/results/EDBOplus_for_B-H_HTE-with-cvt.csv",
            "target_columns": ["yield_collected_values", "cost_collected_values"],
            "direction_tags": ["max", "min"],
            "range_tags": [[0, 100], [0, 0.5]],
        },
        "Gryffin": {
            "results_path": "compare_mothods/gryffin/results/merged_Gryffin_for_B-H_HTE.csv",
            "target_columns": ["yield", "cost"],
            "direction_tags": ["max", "min"],
            "range_tags": [[0, 100], [0, 0.5]],
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
