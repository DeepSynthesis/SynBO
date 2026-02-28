import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import glob
import copy
import json
import re
from matplotlib import ticker


def load_benchmark_results(results_dir="./results"):
    """
    加载所有基准测试结果数据

    Args:
        results_dir: 结果文件夹路径

    Returns:
        dict: {experiment_name: dict} 映射，包含config和所有round的数据
    """
    results_dict = {}
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Results directory {results_dir} does not exist!")
        return results_dict

    # 查找所有实验文件夹 (multiple_* 格式)
    for exp_dir in results_path.glob("multiple_*"):
        if exp_dir.is_dir():
            try:
                # 读取config.json
                config_file = exp_dir / "config.json"
                if not config_file.exists():
                    print(f"No config.json found in {exp_dir}")
                    continue

                with open(config_file, "r") as f:
                    config = json.load(f)

                experiment_name = config.get("experiment_name", exp_dir.name)

                # 读取所有round的CSV文件
                round_data = {}
                csv_files = list(exp_dir.glob("all_batches_final_round_*.csv"))

                for csv_file in csv_files:
                    # 从文件名提取round号
                    round_match = re.search(r"round_(\d+)\.csv", csv_file.name)
                    if round_match:
                        round_num = int(round_match.group(1))
                        df = pd.read_csv(csv_file)
                        round_data[round_num] = df

                if round_data:
                    results_dict[experiment_name] = {"config": config, "rounds": round_data, "folder": exp_dir.name}
                    total_records = sum(len(df) for df in round_data.values())
                    print(f"Loaded {experiment_name}: {len(round_data)} rounds, {total_records} total records")
                else:
                    print(f"No valid CSV files found in {exp_dir}")

            except Exception as e:
                print(f"Failed to load {exp_dir}: {e}")

    return results_dict


def parse_experiment_config(exp_name, config):
    """
    从配置文件中提取各个参数

    Args:
        exp_name: 实验名称
        config: 配置字典

    Returns:
        dict: 包含各个参数的字典
    """
    result = {
        "experiment_name": exp_name,
        "surrogate_model": None,
        "optimize_method": None,
        "acq_function": None,
        "evolution_method": None,
        "sampling_method": None,
    }

    # 从optimization_settings中提取参数
    opt_settings = config.get("optimization_settings", {})

    # 提取基本参数
    result["optimize_method"] = opt_settings.get("optimize_method")
    result["sampling_method"] = opt_settings.get("sampling_method")

    # 从kwargs中提取详细参数
    kwargs = opt_settings.get("kwargs", {})
    result["surrogate_model"] = kwargs.get("surrogate_model")
    result["acq_function"] = kwargs.get("acq_func")
    result["evolution_method"] = kwargs.get("evolution_method")

    return result


def plot_comparative_optimization_curves(
    results_dict, target_columns, direction_tags, range_tags, fixed_params, vary_param, output_dir="./benchmark_plots"
):
    """
    绘制横向对比的优化曲线

    Args:
        results_dict: 实验结果字典
        target_columns: 目标列名列表
        direction_tags: 优化方向列表 ['max', 'min', ...]
        range_tags: Y轴范围列表 [(min, max), ...]
        fixed_params: 固定参数字典，如 {"sampling_method": "lhs", "optimize_method": "default_BO"}
        vary_param: 变化的参数名，如 "surrogate_model"
        output_dir: 输出目录
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # 筛选符合条件的实验
    filtered_results = {}
    for exp_name, exp_data in results_dict.items():
        config = exp_data["config"]
        exp_params = parse_experiment_config(exp_name, config)

        # 检查固定参数是否匹配
        match = True
        for key, value in fixed_params.items():
            if exp_params.get(key) != value:
                match = False
                break

        if match and exp_params.get(vary_param) is not None:
            vary_value = exp_params[vary_param]
            if vary_value not in filtered_results:
                filtered_results[vary_value] = []
            # 将所有rounds的数据合并
            for round_num, df in exp_data["rounds"].items():
                filtered_results[vary_value].append(df)

    if not filtered_results:
        print(f"No experiments found matching fixed params: {fixed_params}")
        return

    # 预处理数据：计算历史最优值
    processed_data = []

    for vary_value, df_list in filtered_results.items():
        for df_idx, df in enumerate(df_list):
            for r_idx in df["round_index"].unique():
                sub_df = df[df["round_index"] == r_idx].sort_values("batch_index").copy()

                # 计算历史最优值
                for col, direction in zip(target_columns, direction_tags):
                    col_name = f"best_{col}"
                    if direction == "max":
                        sub_df[col_name] = sub_df[col].cummax()
                    elif direction == "min":
                        sub_df[col_name] = sub_df[col].cummin()

                # 添加分组信息
                sub_df[vary_param] = vary_value
                sub_df["experiment_id"] = f"{vary_value}_{df_idx}_{r_idx}"
                processed_data.append(sub_df)

    if not processed_data:
        print("No data to process")
        return

    plot_df = pd.concat(processed_data, ignore_index=True)

    # 绘图
    n_cols = len(target_columns)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5), constrained_layout=True)
    if n_cols == 1:
        axes = [axes]

    colors = plt.cm.Set1(np.linspace(0, 1, len(filtered_results)))

    for i, col in enumerate(target_columns):
        ax = axes[i]
        direction = direction_tags[i]
        y_range = range_tags[i]
        col_best_name = f"best_{col}"

        # 绘制每个变化参数值的曲线
        for j, vary_value in enumerate(sorted(filtered_results.keys())):
            subset = plot_df[plot_df[vary_param] == vary_value]
            if not subset.empty:
                sns.lineplot(
                    data=subset,
                    x="batch_index",
                    y=col_best_name,
                    ax=ax,
                    label=f"{vary_param}={vary_value}",
                    color=colors[j],
                    linewidth=2.5,
                    alpha=0.8,
                )

        # 设置标题和轴标签
        arrow = "↗" if direction == "max" else "↘"
        ax.set_title(f"{col} ({arrow} {direction})", fontsize=14, fontweight="bold")
        ax.set_xlabel("Batch Index", fontsize=12)
        ax.set_ylabel(f"Best {col}", fontsize=12)

        # 设置Y轴范围
        if y_range and y_range[0] is not None and y_range[1] is not None:
            margin = (y_range[1] - y_range[0]) * 0.05
            ax.set_ylim(y_range[0] - margin, y_range[1] + margin)

        # 网格和图例
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    # 主标题
    fixed_str = ", ".join([f"{k}={v}" for k, v in fixed_params.items()])
    fig.suptitle(f"Optimization Curves Comparison\nFixed: {fixed_str}", fontsize=16, fontweight="bold")

    # 保存图片
    filename = f"comparison_{vary_param}_fixed_" + "_".join([f"{k}_{v}" for k, v in fixed_params.items()]) + ".png"
    save_path = output_path / filename
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Comparative optimization curves saved: {save_path}")


def plot_comparative_final_performance(
    results_dict, target_columns, direction_tags, range_tags, fixed_params, vary_param, output_dir="./benchmark_plots"
):
    """
    绘制横向对比的最终性能箱型图

    Args:
        results_dict: 实验结果字典
        target_columns: 目标列名列表
        direction_tags: 优化方向列表
        range_tags: Y轴范围列表
        fixed_params: 固定参数字典
        vary_param: 变化的参数名
        output_dir: 输出目录
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # 筛选符合条件的实验
    filtered_results = {}
    for exp_name, exp_data in results_dict.items():
        config = exp_data["config"]
        exp_params = parse_experiment_config(exp_name, config)

        # 检查固定参数是否匹配
        match = True
        for key, value in fixed_params.items():
            if exp_params.get(key) != value:
                match = False
                break

        if match and exp_params.get(vary_param) is not None:
            vary_value = exp_params[vary_param]
            if vary_value not in filtered_results:
                filtered_results[vary_value] = []
            # 将所有rounds的数据合并
            for round_num, df in exp_data["rounds"].items():
                filtered_results[vary_value].append(df)

    if not filtered_results:
        print(f"No experiments found matching fixed params: {fixed_params}")
        return

    # 提取最终最佳值
    final_performance_data = []

    for vary_value, df_list in filtered_results.items():
        for df in df_list:
            for r_idx in df["round_index"].unique():
                sub_df = df[df["round_index"] == r_idx]
                record = {vary_param: vary_value, "round_index": r_idx}

                for col, direction in zip(target_columns, direction_tags):
                    if direction == "max":
                        best_val = sub_df[col].max()
                    elif direction == "min":
                        best_val = sub_df[col].min()
                    record[col] = best_val

                final_performance_data.append(record)

    if not final_performance_data:
        print("No performance data to plot")
        return

    performance_df = pd.DataFrame(final_performance_data)

    # 转换为长格式以便绘图
    melted_df = performance_df.melt(
        id_vars=[vary_param, "round_index"], value_vars=target_columns, var_name="Target", value_name="Best Value"
    )

    # 绘图
    n_cols = len(target_columns)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5), constrained_layout=True)
    if n_cols == 1:
        axes = [axes]

    colors = sns.color_palette("Set1", n_colors=len(filtered_results))

    for i, col in enumerate(target_columns):
        ax = axes[i]
        direction = direction_tags[i]
        y_range = range_tags[i]

        # 筛选当前目标的数据
        col_data = melted_df[melted_df["Target"] == col]

        # 绘制箱型图
        sns.boxplot(data=col_data, x=vary_param, y="Best Value", ax=ax, palette=colors, linewidth=1.5, boxprops=dict(alpha=0.7))

        # 叠加散点图
        sns.stripplot(data=col_data, x=vary_param, y="Best Value", ax=ax, size=6, alpha=0.8, color="black", jitter=True)

        # 设置标题和轴标签
        arrow = "↗" if direction == "max" else "↘"
        ax.set_title(f"{col} ({arrow} {direction})", fontsize=14, fontweight="bold")
        ax.set_xlabel(vary_param.replace("_", " ").title(), fontsize=12)
        ax.set_ylabel(f"Best {col}", fontsize=12)

        # 设置Y轴范围
        if y_range and y_range[0] is not None and y_range[1] is not None:
            margin = (y_range[1] - y_range[0]) * 0.05
            ax.set_ylim(y_range[0] - margin, y_range[1] + margin)

        # 旋转x轴标签以避免重叠
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        ax.grid(True, alpha=0.3)

    # 主标题
    fixed_str = ", ".join([f"{k}={v}" for k, v in fixed_params.items()])
    fig.suptitle(f"Final Performance Comparison\nFixed: {fixed_str}", fontsize=16, fontweight="bold")

    # 保存图片
    filename = f"final_performance_{vary_param}_fixed_" + "_".join([f"{k}_{v}" for k, v in fixed_params.items()]) + ".png"
    save_path = output_path / filename
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Final performance comparison saved: {save_path}")


def generate_all_comparisons(results_dict, target_columns, direction_tags, range_tags, output_dir="./benchmark_plots"):
    """
    生成所有可能的参数对比图

    Args:
        results_dict: 实验结果字典
        target_columns: 目标列名列表
        direction_tags: 优化方向列表
        range_tags: Y轴范围列表
        output_dir: 输出目录
    """
    # 定义可能的参数组合
    comparison_configs = [
        # 比较优化方法
        {
            "fixed_params": {"sampling_method": "lhs"},
            "vary_param": "optimize_method"
        },
        # 比较代理模型（对于default_BO）
        {
            "fixed_params": {"optimize_method": "default_BO", "sampling_method": "kmeans", "acq_function": "EHVI"},
            "vary_param": "surrogate_model",
        },
        # 比较获取函数（对于default_BO）
        {
            "fixed_params": {"optimize_method": "default_BO", "sampling_method": "lhs", "surrogate_model": "GP"},
            "vary_param": "acq_function"
        },
        # 比较采样方法
        {
            "fixed_params": {"optimize_method": "default_BO", "surrogate_model": "GP", "acq_function": "EHVI"},
            "vary_param": "sampling_method"
        },
        # 比较进化方法（对于evolution）
        {
            "fixed_params": {"optimize_method": "evolution", "sampling_method": "lhs", "surrogate_model": "GP"},
            "vary_param": "evolution_method"
        }
    ]

    for config in comparison_configs:
        try:
            print(f"\nGenerating comparison for {config['vary_param']} with fixed params: {config['fixed_params']}")

            # 生成优化曲线对比图
            plot_comparative_optimization_curves(
                results_dict, target_columns, direction_tags, range_tags, config["fixed_params"], config["vary_param"], output_dir
            )

            # 生成最终性能对比图
            plot_comparative_final_performance(
                results_dict, target_columns, direction_tags, range_tags, config["fixed_params"], config["vary_param"], output_dir
            )

        except Exception as e:
            print(f"Error generating comparison for {config}: {e}")


def main():
    """
    主函数：加载数据并生成所有对比图
    """
    # 配置参数 - 根据实际数据调整
    target_columns = ["yield", "cost"]  # 根据CSV文件中的列名
    direction_tags = ["max", "min"]  # yield最大化，cost最小化
    range_tags = [(0, 100), (0, 0.5)]  # 根据实际数据范围调整

    results_dir = "./results"
    output_dir = "./benchmark_plots"

    # 加载基准测试结果
    print("Loading benchmark results...")
    results_dict = load_benchmark_results(results_dir)

    if not results_dict:
        print("No results found!")
        return

    print(f"Loaded {len(results_dict)} experiments")

    # 生成所有对比图
    print("Generating comparison plots...")
    generate_all_comparisons(results_dict, target_columns, direction_tags, range_tags, output_dir)

    print(f"All plots saved to {output_dir}")


if __name__ == "__main__":
    main()
