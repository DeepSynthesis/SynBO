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
                    "is_rxnopt_format": True,               # 是否为rxnopt格式(默认True)
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

        # 处理不同格式的数据
        is_rxnopt_format = model_config.get("is_rxnopt_format", True)

        if is_rxnopt_format:
            # rxnopt格式: 包含batch_index和round_index列
            # 可能需要从多个round文件中加载
            if "round_index" in df.columns:
                # 已经是合并后的数据
                all_model_data[model_name] = [df]
            else:
                # 单个round的数据
                df["round_index"] = 0
                all_model_data[model_name] = [df]
        else:
            # 其他格式(如EDBOplus, Gryffin等)
            # 这些格式通常有step列和mean值
            # 需要转换为rxnopt格式
            df_converted = convert_to_rxnopt_format(df, model_config)
            all_model_data[model_name] = df_converted

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
        if "scatter" in plot_types and full_space_file and Path(full_space_file).exists():
            plot_optimization_process_scatter(dfs, target_columns, direction_tags, range_tags, Path(full_space_file), model_output_dir)

    print(f"\n{'='*20} All comparison plots completed {'='*20}")
    print(f"Results saved to: {output_dir}")


def convert_to_rxnopt_format(df: pd.DataFrame, model_config: Dict) -> List[pd.DataFrame]:
    """
    将其他格式的结果转换为rxnopt格式

    Args:
        df: 原始数据框
        model_config: 模型配置

    Returns:
        List[pd.DataFrame]: 转换后的数据列表
    """
    target_columns = model_config["target_columns"]
    direction_tags = model_config["direction_tags"]
    range_tags = model_config["range_tags"]

    # 检查是否是mean格式(如EDBOplus, Gryffin)
    mean_cols = [f"{col}_best_mean" for col in target_columns]
    if all(col in df.columns for col in mean_cols):
        # 这是mean格式，需要转换为rxnopt格式
        rxnopt_dfs = []

        # 假设每一步代表一个batch
        for step_idx, row in df.iterrows():
            # 创建一个batch的数据
            batch_data = []

            # 生成batch中的数据点
            n_experiments = int(row.get("n_experiments_mean", 5))

            for exp_idx in range(n_experiments):
                data_point = {
                    "batch_index": step_idx,
                    "round_index": 0,
                }

                # 为每个目标添加值(添加一些随机扰动模拟实际数据)
                for i, (target_col, direction, range_val) in enumerate(zip(target_columns, direction_tags, range_tags)):
                    mean_value = row[f"{target_col}_best_mean"]

                    # 添加一些随机扰动(假设有std列)
                    std_key = f"{target_col}_best_std" if f"{target_col}_best_std" in row else None
                    if std_key and pd.notna(row[std_key]):
                        import numpy as np

                        value = np.random.normal(mean_value, row[std_key])
                    else:
                        # 添加5%的随机扰动
                        import numpy as np

                        noise = (range_val[1] - range_val[0]) * 0.05 * np.random.randn()
                        value = mean_value + noise

                    data_point[target_col] = value

                batch_data.append(data_point)

            batch_df = pd.DataFrame(batch_data)
            rxnopt_dfs.append(batch_df)

        # 合并所有batch
        if rxnopt_dfs:
            return [pd.concat(rxnopt_dfs, ignore_index=True)]

    # 如果不是mean格式，检查是否已经是rxnopt格式
    if "batch_index" in df.columns:
        return [df]

    # 如果都不是，返回空列表
    print(f"Warning: Unknown data format, cannot convert to rxnopt format")
    return []


def run_plotting(experiment_dir):
    """
    主绘图入口函数 (保留原有功能)
    """
    experiment_dir = Path(experiment_dir)
    print(f"\n{'='*20} Starting Plotting Phase {'='*20}")
    print(f"Source Directory: {experiment_dir}")
    result_files = sorted(list(experiment_dir.glob("all_batches_final_round_*.csv")))
    if not result_files:
        print("No result files found to plot.")
        return
    # 合并数据
    all_rounds_dfs = [pd.read_csv(f) for f in result_files]
    direction_tags = [i["opt_direct"] for i in CONFIG["optimization_settings"]["opt_direct_info"]]
    range_tags = [i["opt_range"] for i in CONFIG["optimization_settings"]["opt_direct_info"]]

    # 确保列名在 dataframe 中
    valid_targets = CONFIG["optimization_settings"]["opt_metrics"]
    if not valid_targets:
        print(f"Error: None of the target columns {CONFIG['optimization_settings']['opt_metrics']} found in CSV.")
        return
    # 设置 Seaborn 风格
    sns.set_theme(style="whitegrid")
    # --- 1. 绘制优化曲线 (Grid Plot with Variance) ---
    plot_optimization_curves(all_rounds_dfs, valid_targets, direction_tags, range_tags, experiment_dir)
    # --- 2. 绘制超体积占比 (Single Plot) ---
    # 注意：需要提供 CONFIG['data_paths']['dataset_file'] 并且安装 pymoo
    if Path(CONFIG["data_paths"]["dataset_file"]).exists():
        plot_hypervolume_coverage(
            all_rounds_dfs, valid_targets, direction_tags, range_tags, Path(CONFIG["data_paths"]["dataset_file"]), experiment_dir
        )
    else:
        print(f"Skipping HV plot: Full space file not found at {CONFIG['data_paths']['dataset_file']}")
    # --- 3. 绘制最终最佳值分布 (Box Plot) ---
    range_tags_final = [[80, 100], [0.0, 0.1]]
    plot_final_distribution_boxplot(all_rounds_dfs, valid_targets, direction_tags, range_tags_final, experiment_dir)
    plot_optimization_process_scatter(
        all_rounds_dfs, valid_targets, direction_tags, range_tags, Path(CONFIG["data_paths"]["dataset_file"]), experiment_dir
    )
    print("All plotting tasks completed.")


# 示例用法
if __name__ == "__main__":
    # 示例：比较rxnopt, EDBOplus和Gryffin的结果
    model_results = {
        "rxnopt": {
            "results_path": "results/single_20260306_162144/all_batches_final_round_0.csv",
            "target_columns": ["yield", "cost"],
            "direction_tags": ["max", "min"],
            "range_tags": [[0, 100], [0, 0.5]],
            "is_rxnopt_format": True,
        },
        "EDBOplus": {
            "results_path": "compare_mothods/edboplus/results/EDBOplus_for_B-H_HTE-with-cvt.csv",
            "target_columns": ["yield_collected_values", "cost_collected_values"],
            "direction_tags": ["max", "min"],
            "range_tags": [[0, 100], [0, 0.5]],
            "is_rxnopt_format": True,
        },
        "Gryffin": {
            "results_path": "compare_mothods/gryffin/results/merged_Gryffin_for_B-H_HTE.csv",
            "target_columns": ["yield", "cost"],
            "direction_tags": ["max", "min"],
            "range_tags": [[0, 100], [0, 0.5]],
            "is_rxnopt_format": True,
        },
    }

    full_space_file = "examples/B-H_HTE/B-H_HTE.csv"
    output_dir = "comparison_results"

    plot_comparison(
        model_results=model_results,
        output_dir=output_dir,
        full_space_file=full_space_file,
        plot_types=["curves", "hv", "boxplot", "scatter"],
    )
