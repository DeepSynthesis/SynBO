import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


def plot_optimization_curves(model_data, target_columns, direction_tags, range_tags, experiment_dir):
    """
    Plot optimization curves for multiple models.
    
    Args:
        model_data: Dictionary with model names as keys and list of DataFrames as values
        target_columns: List of target column names
        direction_tags: Optimization direction list
        range_tags: Theoretical range list
        experiment_dir: Image save directory
    """
    if not (len(target_columns) == len(direction_tags) == len(range_tags)):
        raise ValueError("target_columns, direction_tags, and range_tags must have same length.")

    all_best_records = []
    all_actual_records = []

    for model_name, dfs in model_data.items():
        for df_idx, df in enumerate(dfs):
            df = df.sort_values("batch").copy()

            for col, direction in zip(target_columns, direction_tags):
                # Group by batch and find the best value for each batch
                if direction == "max":
                    batch_best = df.groupby("batch")[col].max()
                elif direction == "min":
                    batch_best = df.groupby("batch")[col].min()
                else:
                    raise ValueError(f"Unknown direction '{direction}' for column '{col}'. Use 'max' or 'min'.")

                # Calculate cumulative best across batches (current and all previous batches)
                if direction == "max":
                    cumulative_best = batch_best.cummax()
                elif direction == "min":
                    cumulative_best = batch_best.cummin()

                # Add cumulative best records (best value up to and including current batch)
                for batch_idx, best_value in cumulative_best.items():
                    all_best_records.append({"batch": batch_idx, "target": col, "value": best_value, "model": model_name})

                # Add batch best records (best value within each batch only)
                for batch_idx, best_value in batch_best.items():
                    all_actual_records.append({"batch": batch_idx, "target": col, "value": best_value, "model": model_name})

    best_df = pd.DataFrame(all_best_records)
    actual_df = pd.DataFrame(all_actual_records)

    n_cols = len(target_columns)
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 5), constrained_layout=True)

    if n_cols == 1:
        axes = [axes]

    for i, col in enumerate(target_columns):
        ax = axes[i]
        direction = direction_tags[i]
        y_range = range_tags[i]
        col_best_data = best_df[best_df["target"] == col]
        col_actual_data = actual_df[actual_df["target"] == col]

        # Lineplot with model hue for cumulative best values
        sns.lineplot(
            data=col_best_data,
            x="batch",
            y="value",
            hue="model",
            ax=ax,
            errorbar=("ci", 95),
            linewidth=2.5,
            marker="o",
            markersize=6,
            legend="full",
            zorder=10,
        )

        # Boxplot with model hue for batch best distribution
        sns.boxplot(
            data=col_actual_data,
            x="batch",
            y="value",
            hue="model",
            ax=ax,
            width=0.6,
            fliersize=0,
            linewidth=1.0,
            legend=False,
            zorder=5,
        )

        ax.set_title(f"Optimization Trace: {col}", fontsize=16, fontname="Arial", fontweight="bold")
        ax.set_xlabel("Batch Iteration", fontsize=14, fontname="Arial", fontweight="bold")

        ax.set_ylabel(f"Best Found {col}", fontsize=14, fontname="Arial", fontweight="bold")

        ax.tick_params(axis="both", which="major", labelsize=12, width=1.5, length=6)
        ax.tick_params(axis="both", which="minor", width=1, length=4)

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname("Arial")

        if y_range is not None:
            if isinstance(y_range, (tuple, list)) and len(y_range) == 2:
                y_min, y_max = y_range
                current_ylim = ax.get_ylim()
                new_min = y_min - (y_max - y_min) * 0.1 if y_min is not None else current_ylim[0]
                new_max = y_max + (y_max - y_min) * 0.1 if y_max is not None else current_ylim[1]
                ax.set_ylim(new_min, new_max)

        ax.grid(True, linestyle="--", alpha=0.6, linewidth=0.8)
        ax.spines["top"].set_linewidth(1.2)
        ax.spines["right"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)
        ax.spines["left"].set_linewidth(1.2)

        ax.legend(loc="best", fontsize=11, framealpha=0.9)

    save_path = experiment_dir / "plot_1_optimization_curves.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot 1 saved: {save_path}")


def plot_hypervolume_coverage(model_data, target_columns, direction_tags, range_tags, full_space_file, experiment_dir):
    """
    Plot hypervolume coverage percentage across all models and runs.

    Similar to plot_optimization_curves:
    1. Boxplot: HV distribution for each batch (across multiple rounds)
    2. Lineplot with errorbar: Cumulative best HV (current and all previous batches)

    Key logic:
    1. Normalize all data to [0, 1] using range_tags.
    2. Convert all objectives to minimization (pymoo HV default).
       - min: normalized_value
       - max: 1.0 - normalized_value
    3. Reference point set to [1.1, 1.1, ...] (slightly above worst value 1.0).

    Args:
        model_data: Dictionary with model names as keys and list of DataFrames as values
        target_columns: List of target column names.
        direction_tags: Optimization direction list, e.g., ['max', 'min', 'max'].
        range_tags: Theoretical range list, List[tuple(min, max)], for normalization.
        full_space_file: Full space data file path.
        experiment_dir: Image save directory (Path object or string).
    """
    from pymoo.indicators.hv import HV

    if len(target_columns) != len(direction_tags) or len(target_columns) != len(range_tags):
        print("Error: The length of 'target_columns', 'direction_tags', and 'range_tags' must be same.")
        return

    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    if str(full_space_file).endswith(".csv"):
        full_df = pd.read_csv(full_space_file)
    else:
        full_df = pd.read_excel(full_space_file)

    full_data_raw = full_df[target_columns].values

    def normalize_and_transform(data, ranges, directions):
        """
        1. Normalize to [0, 1] using range_tags.
        2. Convert to minimization problem using direction_tags.
        """
        data = np.array(data)
        transformed = np.zeros_like(data, dtype=float)

        for i, (col_min, col_max) in enumerate(ranges):
            col_data = data[:, i]
            if col_max == col_min:
                norm_col = np.full_like(col_data, 1.0)
            else:
                norm_col = (col_data - col_min) / (col_max - col_min)

            norm_col = np.clip(norm_col, 0.0, 1.0)

            direction = directions[i]
            if direction == "min":
                transformed[:, i] = norm_col
            elif direction == "max":
                transformed[:, i] = 1.0 - norm_col
            else:
                print(f"Warning: Unknown direction '{direction}', assuming 'max'.")
                transformed[:, i] = 1.0 - norm_col

        return transformed

    full_data_norm = normalize_and_transform(full_data_raw, range_tags, direction_tags)
    ref_point = np.array([1.1] * len(target_columns))
    ind = HV(ref_point=ref_point)
    total_hv = ind(full_data_norm)

    if total_hv == 0:
        print("Warning: Total Hypervolume is 0. Check your range_tags or data.")
        return

    print(f"Total Theoretical HV (Normalized): {total_hv:.4f}")

    all_best_records = []
    all_actual_records = []

    for model_name, dfs in model_data.items():
        for df in dfs:
            df = df.sort_values("batch").copy()
            round_data_raw = df[target_columns].values
            round_data_norm = normalize_and_transform(round_data_raw, range_tags, direction_tags)

            # Calculate HV for each batch (HV of all samples up to and including that batch)
            batch_hv_values = {}
            batch_indices = sorted(df["batch"].unique())

            for batch_idx in batch_indices:
                # Get all samples up to and including this batch
                batch_samples = df[df["batch"] <= batch_idx]
                batch_data_raw = batch_samples[target_columns].values
                batch_data_norm = normalize_and_transform(batch_data_raw, range_tags, direction_tags)

                # Calculate HV for this batch
                current_hv = ind(batch_data_norm)
                hv_percentage = (current_hv / total_hv) * 100

                batch_hv_values[batch_idx] = hv_percentage

            # Convert to series for cummax and groupby
            batch_hv_series = pd.Series(batch_hv_values)

            # Calculate cumulative best HV (hypervolume should be maximized)
            cumulative_best_hv = batch_hv_series.cummax()

            # Add cumulative best records (best HV up to and including current batch)
            for batch_idx, hv_value in cumulative_best_hv.items():
                all_best_records.append({"batch": batch_idx, "value": hv_value, "model": model_name})

            # Add batch HV records (HV at each batch)
            for batch_idx, hv_value in batch_hv_series.items():
                all_actual_records.append({"batch": batch_idx, "value": hv_value, "model": model_name})

    best_df = pd.DataFrame(all_best_records)
    actual_df = pd.DataFrame(all_actual_records)

    if best_df.empty or actual_df.empty:
        print("No data to plot.")
        return

    plt.figure(figsize=(10, 6))

    # Lineplot with model hue for cumulative best HV
    sns.lineplot(
        data=best_df,
        x="batch",
        y="value",
        hue="model",
        errorbar=("ci", 95),
        linewidth=2.5,
        marker="o",
        markersize=6,
        legend="full",
        zorder=10,
    )

    # Boxplot with model hue for batch HV distribution
    sns.boxplot(
        data=actual_df,
        x="batch",
        y="value",
        hue="model",
        width=0.6,
        fliersize=0,
        linewidth=1.0,
        legend=False,
        zorder=5,
    )

    plt.title(
        f"Hypervolume Coverage",
        fontsize=16,
        fontname="Arial",
        fontweight="bold",
    )
    plt.xlabel("Batch Iteration", fontsize=14, fontname="Arial", fontweight="bold")
    plt.ylabel("Hypervolume Coverage (%)", fontsize=14, fontname="Arial", fontweight="bold")

    plt.tick_params(axis="both", which="major", labelsize=12, width=1.5, length=6)
    plt.tick_params(axis="both", which="minor", width=1, length=4)

    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        label.set_fontname("Arial")

    plt.ylim(0, 105)
    plt.grid(True, linestyle="--", alpha=0.6, linewidth=0.8)
    ax = plt.gca()
    ax.spines["top"].set_linewidth(1.2)
    ax.spines["right"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["left"].set_linewidth(1.2)

    ax.legend(loc="lower right", fontsize=11, framealpha=0.9)

    save_path = experiment_dir / "plot_2_hypervolume_coverage.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot 2 saved: {save_path}")


def plot_final_distribution_boxplot(model_data, target_columns, direction_tags, range_tags, experiment_dir):
    """
    绘制美化版箱型图：所有模型最后优化到的最佳值分布。

    Args:
        model_data: Dictionary with model names as keys and list of DataFrames as values
        target_columns: List of target column names
        direction_tags: Optimization direction list
        range_tags: Theoretical range list
        experiment_dir: Image save directory
    """
    # 1. 提取每一个df的最终最佳值
    final_best_records = []
    dir_map = dict(zip(target_columns, direction_tags))
    range_map = dict(zip(target_columns, range_tags))

    for model_name, dfs in model_data.items():
        for df_idx, df in enumerate(dfs):
            record = {"run_index": df_idx, "model": model_name}
            for col in target_columns:
                direction = dir_map[col]
                if direction == "max":
                    best_val = df[col].max()
                elif direction == "min":
                    best_val = df[col].min()
                else:
                    raise Exception(f"Unknown direction: {direction}")
                record[col] = best_val
            final_best_records.append(record)

    best_df = pd.DataFrame(final_best_records)

    # 2. 转换数据格式
    melted_df = best_df.melt(id_vars=["run_index", "model"], value_vars=target_columns, var_name="Target", value_name="Best Value")

    # 3. 绘图初始化
    n_cols = len(target_columns)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5), constrained_layout=True)

    if n_cols == 1:
        axes = [axes]

    for i, col in enumerate(target_columns):
        ax = axes[i]
        direction = direction_tags[i]
        y_range = range_tags[i]

        col_data = melted_df[melted_df["Target"] == col]

        sns.boxplot(
            data=col_data,
            x="model",
            y="Best Value",
            ax=ax,
            width=0.6,
            fliersize=0,
            linewidth=1.0,
            zorder=5,
        )

        sns.stripplot(
            data=col_data,
            x="model",
            y="Best Value",
            ax=ax,
            size=6,
            jitter=0.2,
            alpha=0.7,
            edgecolor="white",
            linewidth=1,
            zorder=10,
        )

        ax.set_title(col, fontsize=16, fontname="Arial", fontweight="bold")
        ax.set_xlabel("", fontsize=14, fontname="Arial", fontweight="bold")

        ax.set_ylabel(f"Best Found {col}", fontsize=14, fontname="Arial", fontweight="bold")

        ax.tick_params(axis="both", which="major", labelsize=12, width=1.5, length=6)
        ax.tick_params(axis="both", which="minor", width=1, length=4)

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname("Arial")

        if y_range is not None:
            if isinstance(y_range, (tuple, list)) and len(y_range) == 2:
                y_min, y_max = y_range
                current_ylim = ax.get_ylim()
                new_min = y_min - (y_max - y_min) * 0.1 if y_min is not None else current_ylim[0]
                new_max = y_max + (y_max - y_min) * 0.1 if y_max is not None else current_ylim[1]
                ax.set_ylim(new_min, new_max)

        ax.grid(True, linestyle="--", alpha=0.6, linewidth=0.8)
        ax.spines["top"].set_linewidth(1.2)
        ax.spines["right"].set_linewidth(1.2)
        ax.spines["bottom"].set_linewidth(1.2)
        ax.spines["left"].set_linewidth(1.2)

        ax.legend(loc="best", fontsize=11, framealpha=0.9)

    save_path = experiment_dir / "plot_3_best_value_distribution.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot 3 saved: {save_path}")


def plot_optimization_process_scatter(model_data, target_columns, direction_tags, range_tags, full_space_file, experiment_dir):
    """
    绘制优化过程散点图：
    1. 背景：利用 full_space_file 计算全空间真实帕累托前沿，连线+绘制节点
    2. 前景：绘制每个模型的帕累托前沿（不用绘制节点，颜色淡一点），所有前沿画在一张图上
    """
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    n_targets = len(target_columns)

    # ==========================================
    # 1. 处理全空间数据 (True Pareto Front)
    # ==========================================
    if str(full_space_file).endswith(".csv"):
        full_df = pd.read_csv(full_space_file)
    else:
        full_df = pd.read_excel(full_space_file)
    full_data_raw = full_df[target_columns].values

    # 准备排序数据 (Pymoo 默认最小化，对 max 目标取反)
    full_sorting_data = full_data_raw.copy()
    for i, direction in enumerate(direction_tags):
        if direction == "max":
            full_sorting_data[:, i] *= -1

    # 全空间非支配排序
    nds = NonDominatedSorting()
    full_fronts = nds.do(full_sorting_data)
    true_pf_indices = full_fronts[0]
    true_pf_data = full_data_raw[true_pf_indices]  # 获取真实的 PF 数据

    # --- 为了画线，按照第一个维度排序 ---
    sorted_indices = np.argsort(true_pf_data[:, 0])
    true_pf_data_sorted = true_pf_data[sorted_indices]

    # ==========================================
    # 2. 处理所有模型的帕累托前沿
    # ==========================================
    all_empirical_pfs = []
    model_names = []

    for model_name, dfs in model_data.items():
        # For each model, combine all its dataframes
        combined_data = None
        for df in dfs:
            if combined_data is None:
                combined_data = df[target_columns].values
            else:
                combined_data = np.vstack([combined_data, df[target_columns].values])
        
        if combined_data is not None:
            # 准备实验数据的排序 (同样处理 max/min)
            exp_sorting_data = combined_data.copy()
            for i, direction in enumerate(direction_tags):
                if direction == "max":
                    exp_sorting_data[:, i] *= -1

            # 实验数据非支配排序
            exp_fronts = nds.do(exp_sorting_data)
            empirical_pf_indices = exp_fronts[0]  # 获取实验数据中的 Rank 0 (非支配解)

            # 筛选出要绘制的前景数据
            empirical_pf_data = combined_data[empirical_pf_indices]
            all_empirical_pfs.append((model_name, empirical_pf_data))
            model_names.append(model_name)

    # ==========================================
    # 3. 开始绘图（所有前沿在同一张图上）
    # ==========================================
    if n_targets == 2:
        _plot_2d_scatter(true_pf_data_sorted, all_empirical_pfs, target_columns, range_tags, direction_tags, experiment_dir)
    elif n_targets == 3:
        pass
    else:
        print(f"Warning: {n_targets} objectives detected. Plotting only the first 2 dimensions for visualization.")
        _plot_2d_scatter(
            true_pf_data_sorted,
            all_empirical_pfs,
            target_columns[:2],
            range_tags[:2],
            direction_tags[:2],
            experiment_dir,
        )


def _plot_2d_scatter(true_pf_sorted, all_empirical_pfs, targets, ranges, directions, save_dir):
    """
    内部辅助函数：绘制 2D 图（所有前沿在同一张图上）
    Args:
        true_pf_sorted: 已按X轴排序的全空间真实帕累托前沿点
        all_empirical_pfs: 所有实验中找到的非支配点列表，每个元素是(model_name, pf_data)
        targets: 目标列名列表
        ranges: 目标范围列表
        directions: 优化方向列表
        save_dir: 保存目录
    """
    plt.figure(figsize=(8, 6))

    # Layer 2: 所有 Empirical Pareto Front (前景 - 仅线条，颜色淡一点)
    for model_idx, (model_name, emp_pf_data) in enumerate(all_empirical_pfs):
        # 为了画线，按照第一个维度排序
        emp_sorted_indices = np.argsort(emp_pf_data[:, 0])
        emp_pf_data_sorted = emp_pf_data[emp_sorted_indices]

        # 使用不同的颜色绘制每个模型的前沿
        color = plt.cm.tab10(model_idx % 10)
        plt.plot(
            emp_pf_data_sorted[:, 0],
            emp_pf_data_sorted[:, 1],
            c=color,
            linestyle="-",
            linewidth=2,
            alpha=0.8,
            label=model_name,
            zorder=5,
        )
    # Layer 1: True Pareto Front (背景 - 线条 + 节点)
    # 绘制线条
    plt.plot(
        true_pf_sorted[:, 0],
        true_pf_sorted[:, 1],
        c="#11668a",
        linestyle="-",
        linewidth=1.5,
        alpha=0.9,
        label="True Pareto Front",
        zorder=0,
    )
    # 绘制节点
    plt.scatter(
        true_pf_sorted[:, 0],
        true_pf_sorted[:, 1],
        c="skyblue",
        edgecolors="k",
        linewidth=0.8,
        s=50,
        alpha=0.9,
        zorder=1,
    )

    # 设置坐标轴范围
    if ranges:
        x_min, x_max = ranges[0]
        y_min, y_max = ranges[1]
        x_padding = (x_max - x_min) * 0.05
        y_padding = (y_max - y_min) * 0.05
        plt.xlim(x_min - x_padding, x_max + x_padding)
        plt.ylim(y_min - y_padding, y_max + y_padding)

    # 标签与标题
    plt.xlabel(f"{targets[0]}", fontsize=14, fontname="Arial", fontweight="bold")
    plt.ylabel(f"{targets[1]}", fontsize=14, fontname="Arial", fontweight="bold")
    plt.title("Optimization Process: True PF vs rxnopt PFs", fontsize=16, fontname="Arial", fontweight="bold")

    # 设置tick参数
    plt.tick_params(axis="both", which="major", labelsize=12, width=1.5, length=6)
    plt.tick_params(axis="both", which="minor", width=1, length=4)

    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        label.set_fontname("Arial")

    # 设置边框和网格
    plt.grid(True, linestyle="--", alpha=0.6, linewidth=0.8)
    ax = plt.gca()
    ax.spines["top"].set_linewidth(1.2)
    ax.spines["right"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["left"].set_linewidth(1.2)

    # Legend
    plt.legend(loc="best", fontsize=11, framealpha=0.9)

    save_path = save_dir / "plot_4_pareto_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot 4 saved: {save_path}")
