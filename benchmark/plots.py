import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


def plot_optimization_curves(dfs, target_columns, direction_tags, range_tags, experiment_dir):
    if not (len(target_columns) == len(direction_tags) == len(range_tags)):
        raise ValueError("target_columns, direction_tags, and range_tags must have same length.")

    all_best_records = []
    all_actual_records = []

    for df_idx, df in enumerate(dfs):
        df = df.sort_values("batch_index").copy()

        for col, direction in zip(target_columns, direction_tags):
            # Group by batch_index and find the best value for each batch
            if direction == "max":
                batch_best = df.groupby("batch_index")[col].max()
            elif direction == "min":
                batch_best = df.groupby("batch_index")[col].min()
            else:
                raise ValueError(f"Unknown direction '{direction}' for column '{col}'. Use 'max' or 'min'.")

            # Calculate cumulative best across batches (current and all previous batches)
            if direction == "max":
                cumulative_best = batch_best.cummax()
            elif direction == "min":
                cumulative_best = batch_best.cummin()

            # Add cumulative best records (best value up to and including current batch)
            for batch_idx, best_value in cumulative_best.items():
                all_best_records.append({"batch_index": batch_idx, "target": col, "value": best_value})

            # Add batch best records (best value within each batch only)
            for batch_idx, best_value in batch_best.items():
                all_actual_records.append({"batch_index": batch_idx, "target": col, "value": best_value})

    best_df = pd.DataFrame(all_best_records)
    actual_df = pd.DataFrame(all_actual_records)

    n_cols = len(target_columns)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5), constrained_layout=True)

    if n_cols == 1:
        axes = [axes]

    for i, col in enumerate(target_columns):
        ax = axes[i]
        direction = direction_tags[i]
        y_range = range_tags[i]

        col_best_data = best_df[best_df["target"] == col]
        col_actual_data = actual_df[actual_df["target"] == col]

        sns.lineplot(
            data=col_best_data,
            x="batch_index",
            y="value",
            ax=ax,
            errorbar=("ci", 95),
            linewidth=3.5,
            marker="o",
            markersize=8,
            label=f"Cumulative Best {col}",
            zorder=10,
        )

        sns.boxplot(
            data=col_actual_data,
            x="batch_index",
            y="value",
            ax=ax,
            width=0.4,
            color="lightgray",
            fliersize=3,
            linewidth=1.5,
            boxprops=dict(alpha=0.6, edgecolor="gray"),
            medianprops=dict(color="red", linewidth=2),
            label="Batch Best Distribution",
            zorder=5,
        )

        ax.set_title(f"Optimization Trace: {col} ({direction})", fontsize=16, fontname="Arial", fontweight="bold")
        ax.set_xlabel("Batch Iteration", fontsize=14, fontname="Arial", fontweight="bold")

        y_label_prefix = "Max" if direction == "max" else "Min"
        ax.set_ylabel(f"Best Found {col} ({y_label_prefix})", fontsize=14, fontname="Arial", fontweight="bold")

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


def plot_hypervolume_coverage(dfs, target_columns, direction_tags, range_tags, full_space_file, experiment_dir):
    """
    Plot hypervolume coverage percentage across all runs.

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
        dfs: List of DataFrames containing experimental results with target_columns and 'batch_index'.
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

    for df in dfs:
        df = df.sort_values("batch_index").copy()
        round_data_raw = df[target_columns].values
        round_data_norm = normalize_and_transform(round_data_raw, range_tags, direction_tags)

        # Calculate HV for each batch (HV of all samples up to and including that batch)
        batch_hv_values = {}
        batch_indices = sorted(df["batch_index"].unique())

        for batch_idx in batch_indices:
            # Get all samples up to and including this batch
            batch_samples = df[df["batch_index"] <= batch_idx]
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
            all_best_records.append({"batch_index": batch_idx, "value": hv_value})

        # Add batch HV records (HV at each batch)
        for batch_idx, hv_value in batch_hv_series.items():
            all_actual_records.append({"batch_index": batch_idx, "value": hv_value})

    best_df = pd.DataFrame(all_best_records)
    actual_df = pd.DataFrame(all_actual_records)

    if best_df.empty or actual_df.empty:
        print("No data to plot.")
        return

    plt.figure(figsize=(10, 6))

    # Lineplot with errorbar for cumulative best HV
    sns.lineplot(
        data=best_df,
        x="batch_index",
        y="value",
        errorbar=("ci", 95),
        linewidth=3.5,
        color="#2ca02c",
        marker="o",
        markersize=8,
        markevery=max(1, len(best_df) // 20),
        label="Cumulative Best HV",
        zorder=10,
    )

    # Boxplot for batch HV distribution
    sns.boxplot(
        data=actual_df,
        x="batch_index",
        y="value",
        width=0.4,
        color="lightgray",
        fliersize=3,
        linewidth=1.5,
        boxprops=dict(alpha=0.6, edgecolor="gray"),
        medianprops=dict(color="red", linewidth=2),
        label="Batch HV Distribution",
        zorder=5,
    )

    plt.title(
        f"Hypervolume Coverage ({len(target_columns)} Objectives)\nDirections: {direction_tags}",
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


def plot_final_distribution_boxplot(dfs, target_columns, direction_tags, range_tags, experiment_dir):
    """
    绘制美化版箱型图：所有 df 最后优化到的最佳值分布。
    
    Args:
        dfs: List of DataFrames containing experimental results
        target_columns: List of target column names
        direction_tags: Optimization direction list
        range_tags: Theoretical range list
        experiment_dir: Image save directory
    """
    # 1. 提取每一个df的最终最佳值
    final_best_records = []
    dir_map = dict(zip(target_columns, direction_tags))
    range_map = dict(zip(target_columns, range_tags))

    for df_idx, df in enumerate(dfs):
        record = {"run_index": df_idx}
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
    melted_df = best_df.melt(id_vars=["run_index"], value_vars=target_columns, var_name="Target", value_name="Best Value")

    # 3. 绘图初始化
    n_cols = len(target_columns)
    fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 5), constrained_layout=True)

    if n_cols == 1:
        axes = [axes]

    for i, col in enumerate(target_columns):
        ax = axes[i]
        direction = direction_tags[i]
        y_range = range_tags[i]

        col_data = melted_df[melted_df["Target"] == col]

        sns.boxplot(
            data=col_data,
            x="Target",
            y="Best Value",
            ax=ax,
            width=0.4,
            color="lightgray",
            fliersize=0,
            linewidth=1.5,
            boxprops=dict(alpha=0.6, edgecolor="gray"),
            medianprops=dict(color="red", linewidth=2),
            label="Distribution",
            zorder=5,
        )

        sns.stripplot(
            data=col_data,
            x="Target",
            y="Best Value",
            ax=ax,
            color="#C44E52",
            size=6,
            jitter=0.2,
            alpha=0.9,
            edgecolor="white",
            linewidth=1,
            zorder=10,
        )

        ax.set_title(col, fontsize=16, fontname="Arial", fontweight="bold")
        ax.set_xlabel("", fontsize=14, fontname="Arial", fontweight="bold")

        y_label_prefix = "Max" if direction == "max" else "Min"
        ax.set_ylabel(f"Best Found {col} ({y_label_prefix})", fontsize=14, fontname="Arial", fontweight="bold")

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


def plot_optimization_process_scatter(dfs, target_columns, direction_tags, range_tags, full_space_file, experiment_dir):
    """
    绘制优化过程散点图 (只显示非支配解)：
    1. 背景：利用 full_space_file 计算全空间真实帕累托前沿，并按顺序连成一条浅蓝色线条。
    2. 前景：仅绘制实验结果中找到的非支配解 (Empirical Pareto Front)，以散点显示，颜色根据 Batch Index 渐变。
    """
    df = df[df["round_index"] == 0]

    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    n_targets = len(target_columns)

    # ==========================================
    # 1. 处理全空间数据 (True Pareto Front - Background Line)
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

    # --- 关键修改：为了画线，必须按照第一个维度排序 ---
    # argsort 返回排序后的索引，保证连线是顺滑的，不会乱窜
    sorted_indices = np.argsort(true_pf_data[:, 0])
    true_pf_data_sorted = true_pf_data[sorted_indices]

    # ==========================================
    # 2. 处理实验数据 (Empirical Pareto Front - Foreground Scatter)
    # ==========================================
    exp_data_raw = df[target_columns].values
    batch_indices_raw = df["batch_index"].values

    # 准备实验数据的排序 (同样处理 max/min)
    exp_sorting_data = exp_data_raw.copy()
    for i, direction in enumerate(direction_tags):
        if direction == "max":
            exp_sorting_data[:, i] *= -1

    # 实验数据非支配排序
    exp_fronts = nds.do(exp_sorting_data)
    empirical_pf_indices = exp_fronts[0]  # 获取实验数据中的 Rank 0 (非支配解)

    # 筛选出要绘制的前景数据
    empirical_pf_data = exp_data_raw[empirical_pf_indices]
    empirical_pf_batches = batch_indices_raw[empirical_pf_indices]

    # ==========================================
    # 3. 开始绘图
    # ==========================================
    if n_targets == 2:
        # 注意这里传入的是排序后的 true_pf_data_sorted
        _plot_2d_scatter(
            true_pf_data_sorted, empirical_pf_data, empirical_pf_batches, target_columns, range_tags, direction_tags, experiment_dir
        )
    elif n_targets == 3:
        pass
    else:
        print(f"Warning: {n_targets} objectives detected. Plotting only the first 2 dimensions for visualization.")
        _plot_2d_scatter(
            true_pf_data_sorted,
            empirical_pf_data,
            empirical_pf_batches,
            target_columns[:2],
            range_tags[:2],
            direction_tags[:2],
            experiment_dir,
        )


def _plot_2d_scatter(true_pf_sorted, emp_pf_data, emp_batches, targets, ranges, directions, save_dir):
    """
    内部辅助函数：绘制 2D 图
    Args:
        true_pf_sorted: 已按X轴排序的全空间真实帕累托前沿点 (用于画线)
        emp_pf_data:    实验中找到的非支配点 (用于画散点)
        emp_batches:    这些非支配点对应的 batch index
    """
    plt.figure(figsize=(10, 8))

    # Layer 1: True Pareto Front (背景 - 线条)
    # 使用 plot 而不是 scatter，并设置 zorder 较低
    plt.plot(
        true_pf_sorted[:, 0],
        true_pf_sorted[:, 1],
        c="skyblue",  # 浅蓝色线条
        linestyle="-",  # 实线
        linewidth=2,  # 线宽
        alpha=0.6,  # 半透明
        label="True Pareto Front",
        zorder=0,  # 放在最底层
    )

    # Layer 2: Empirical Pareto Front (前景 - 散点)
    sc = plt.scatter(
        emp_pf_data[:, 0],
        emp_pf_data[:, 1],
        c=emp_batches,
        cmap="Blues",  # 颜色映射
        edgecolors="k",  # 黑色描边
        linewidth=0.8,
        s=80,  # 大小
        alpha=1.0,  # 不透明
        label="Found Non-Dominated Solutions",
        zorder=10,  # 放在最顶层
    )

    # 添加 Colorbar
    if len(emp_batches) > 0:
        cbar = plt.colorbar(sc)
        cbar.set_label("Found at Batch Index")

    # 设置坐标轴范围
    if ranges:
        x_min, x_max = ranges[0]
        y_min, y_max = ranges[1]
        x_padding = (x_max - x_min) * 0.05
        y_padding = (y_max - y_min) * 0.05
        plt.xlim(x_min - x_padding, x_max + x_padding)
        plt.ylim(y_min - y_padding, y_max + y_padding)

    # 标签与标题
    plt.xlabel(f"{targets[0]} ({directions[0]})")
    plt.ylabel(f"{targets[1]} ({directions[1]})")
    plt.title("Optimization Process: Found Solutions vs True PF Line")

    # Legend
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.5)

    save_path = save_dir / "plot_4_pareto_line_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot 4 saved: {save_path}")


def _plot_3d_scatter(pf_data, exp_data, batch_indices, targets, ranges, directions, save_dir):
    """内部辅助函数：绘制 3D 图"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    # Layer 1: True Pareto Front (背景)
    ax.scatter(
        pf_data[:, 0],
        pf_data[:, 1],
        pf_data[:, 2],
        c="skyblue",
        alpha=0.2,  # 3D 图中点多，透明度设低一点
        s=20,
        label="True Pareto Front",
        depthshade=True,
    )
    # Layer 2: Optimization Process (前景)
    sc = ax.scatter(
        exp_data[:, 0],
        exp_data[:, 1],
        exp_data[:, 2],
        c=batch_indices,
        cmap="Blues",
        edgecolors="k",
        linewidth=0.3,
        s=40,
        alpha=1.0,
        label="Observed Samples",
        depthshade=False,  # 关闭深度阴影，保持颜色准确反映 Batch
    )
    # 添加 Colorbar
    cbar = plt.colorbar(sc, ax=ax, pad=0.1)
    cbar.set_label("Batch Iteration")
    # 设置坐标轴范围
    if ranges:
        ax.set_xlim(ranges[0])
        ax.set_ylim(ranges[1])
        ax.set_zlim(ranges[2])
    # 标签
    ax.set_xlabel(f"{targets[0]} ({directions[0]})")
    ax.set_ylabel(f"{targets[1]} ({directions[1]})")
    ax.set_zlabel(f"{targets[2]} ({directions[2]})")
    ax.set_title("Optimization Process (3D)")
    # 调整视角 (可选)
    ax.view_init(elev=30, azim=45)
    save_path = save_dir / "plot_4_optimization_process_3d.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot 4 saved: {save_path}")
