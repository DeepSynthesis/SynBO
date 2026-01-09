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


def plot_optimization_curves(df, target_columns, direction_tags, range_tags, experiment_dir):
    """
    绘制组图：每个目标值的优化曲线（当前 Batch 之前的历史最优值）。
    使用阴影表示不同 Round 之间的 Variance (95% CI 或 SD)。

    Args:
        df: 包含实验数据的 DataFrame。
        target_columns: 需要绘制的目标列名列表。
        direction_tags: 对应每个目标的优化方向列表，['max', 'min', ...]。
        range_tags: 对应每个目标的 Y 轴范围列表，例如 [(0, 1), (0, 100), None]。如果为 None 则自动缩放。
        experiment_dir: 图片保存路径对象（通常是 pathlib.Path）。
    """
    # 校验输入长度是否一致，防止报错
    if not (len(target_columns) == len(direction_tags) == len(range_tags)):
        raise ValueError("target_columns, direction_tags, and range_tags must have the same length.")
    # 1. 预处理：根据方向计算 CumMax 或 CumMin（历史最优值）
    processed_dfs = []

    # 我们需要对每个 round 单独计算历史最优
    for r_idx in df["round_index"].unique():
        sub_df = df[df["round_index"] == r_idx].sort_values("batch_index").copy()

        for col, direction in zip(target_columns, direction_tags):
            col_name = f"best_{col}"  # 统一命名为 best_ 前缀

            if direction == "max":
                # 越大越好：计算截止到当前 batch 的最大值
                sub_df[col_name] = sub_df[col].cummax()
            elif direction == "min":
                # 越小越好：计算截止到当前 batch 的最小值
                sub_df[col_name] = sub_df[col].cummin()
            else:
                raise ValueError(f"Unknown direction '{direction}' for column '{col}'. Use 'max' or 'min'.")

        processed_dfs.append(sub_df)
    plot_df = pd.concat(processed_dfs, ignore_index=True)
    # 2. 设置绘图布局
    n_cols = len(target_columns)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5), constrained_layout=True)
    if n_cols == 1:
        axes = [axes]  # 确保 axes 是列表以便迭代
    for i, col in enumerate(target_columns):
        ax = axes[i]
        direction = direction_tags[i]
        y_range = range_tags[i]
        col_best_name = f"best_{col}"
        # 使用 seaborn lineplot 自动处理聚合和阴影
        sns.lineplot(data=plot_df, x="batch_index", y=col_best_name, ax=ax, errorbar="sd", linewidth=2)  # 使用标准差作为阴影
        # 设置标题和标签
        ax.set_title(f"Optimization Trace: {col} ({direction})")
        ax.set_xlabel("Batch Iteration")

        # 根据方向设置 Y 轴 Label
        y_label_prefix = "Max" if direction == "max" else "Min"
        ax.set_ylabel(f"Best Found {col} ({y_label_prefix})")

        # 设置 Y 轴范围 (range_tags)
        if y_range is not None:
            # 确保是 tuple 且有两个值
            if isinstance(y_range, (tuple, list)) and len(y_range) == 2:
                y_min, y_max = y_range
                # 如果 min 或 max 是 None，则保持 matplotlib 的自动缩放，否则设置限制
                current_ylim = ax.get_ylim()
                new_min = y_min - (y_max - y_min) * 0.1 if y_min is not None else current_ylim[0]
                new_max = y_max + (y_max - y_min) * 0.1 if y_max is not None else current_ylim[1]
                ax.set_ylim(new_min, new_max)
        ax.grid(True, linestyle="--", alpha=0.6)

    save_path = experiment_dir / "plot_1_optimization_curves.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot 1 saved: {save_path}")


def plot_hypervolume_coverage(df, target_columns, direction_tags, range_tags, full_space_file, experiment_dir):
    """
    绘制单个图：当前 Batch 下的超体积占全空间理论超体积的百分比。

    关键逻辑：
    1. 使用 range_tags 将所有数据归一化到 [0, 1]。
    2. 将所有目标转换为“最小化”问题 (pymoo 的 HV 计算默认假设最小化)。
       - min 目标: normalized_value
       - max 目标: 1.0 - normalized_value
    3. 参考点设为 [1.1, 1.1, ...] (略大于最差值 1.0)。

    Args:
        df: 包含实验结果的 DataFrame，需包含 target_columns, 'round_index', 'batch_index'。
        target_columns: 目标列名列表。
        direction_tags: 优化方向列表，例如 ['max', 'min', 'max']。
        range_tags: 理论范围列表，List[tuple(min, max)]，用于归一化。
        full_space_file: 全空间数据文件路径。
        experiment_dir: 图片保存目录 (Path 对象或字符串)。
    """
    try:
        from pymoo.indicators.hv import HV
    except ImportError:
        print("Error: 'pymoo' library is required for Hypervolume calculation. Run 'pip install pymoo'.")
        return

    # 0. 基础检查
    if len(target_columns) != len(direction_tags) or len(target_columns) != len(range_tags):
        print("Error: The length of 'target_columns', 'direction_tags', and 'range_tags' must be the same.")
        return

    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # 1. 读取全空间数据
    if str(full_space_file).endswith(".csv"):
        full_df = pd.read_csv(full_space_file)
    else:
        full_df = pd.read_excel(full_space_file)

    full_data_raw = full_df[target_columns].values

    # === 核心处理函数：归一化 + 最小化转换 ===
    def normalize_and_transform(data, ranges, directions):
        """
        1. 根据 range_tags 归一化到 [0, 1]。
        2. 根据 direction_tags 统一转换为最小化问题。
        """
        data = np.array(data)
        transformed = np.zeros_like(data, dtype=float)

        for i, (col_min, col_max) in enumerate(ranges):
            col_data = data[:, i]
            # 1. 归一化 (Normalization)
            if col_max == col_min:
                # 避免除零，如果最大最小一样，设为 0.5 或者 1.0
                norm_col = np.full_like(col_data, 1.0)
            else:
                norm_col = (col_data - col_min) / (col_max - col_min)

            # 确保归一化后的数据不越界（针对实验数据可能略微超出理论范围的情况）
            norm_col = np.clip(norm_col, 0.0, 1.0)

            # 2. 方向转换 (Transformation to Minimization)
            direction = directions[i]
            if direction == "min":
                # 越小越好，归一化后 0 是最好，1 是最差 -> 保持不变
                transformed[:, i] = norm_col
            elif direction == "max":
                # 越大越好，归一化后 1 是最好，0 是最差
                # 转换为最小化：(1 - value)，此时 0 变为最好，1 变为最差
                transformed[:, i] = 1.0 - norm_col
            else:
                print(f"Warning: Unknown direction '{direction}', assuming 'max'.")
                transformed[:, i] = 1.0 - norm_col

        return transformed

    # 2. 准备数据和 HV 计算器
    # 转换全空间数据
    full_data_norm = normalize_and_transform(full_data_raw, range_tags, direction_tags)

    # 设置参考点
    # 因为数据已经归一化并统一转为最小化问题，值的范围是 [0, 1]
    # 参考点应该比最差值 (1.0) 稍微大一点，以确保边界点被计算进去
    ref_point = np.array([1.1] * len(target_columns))

    # 初始化 HV 指标计算器
    ind = HV(ref_point=ref_point)

    # 计算全空间的理论最大超体积 (Total Theoretical Hypervolume)
    # 注意：pymoo 的 ind() 会自动处理非支配排序，计算输入点集构成的帕累托前沿的体积
    total_hv = ind(full_data_norm)

    if total_hv == 0:
        print("Warning: Total Hypervolume is 0. Check your range_tags or data.")
        return

    print(f"Total Theoretical HV (Normalized): {total_hv:.4f}")

    # 3. 计算每一轮、每一个 Batch 的 HV
    hv_records = []

    # 遍历每一轮实验 (Round)
    for r_idx in sorted(df["round_index"].unique()):
        sub_df = df[df["round_index"] == r_idx].sort_values("batch_index")

        # 提取当前 round 的所有原始数据
        round_data_raw = sub_df[target_columns].values

        # 转换当前 round 数据 (使用同样的规则)
        round_data_norm = normalize_and_transform(round_data_raw, range_tags, direction_tags)

        # 逐步增加样本点计算 HV (模拟实验过程)
        # 注意：这里我们计算的是“当前已发现的所有点”构成的超体积
        for i in range(len(round_data_norm)):
            # 取出从开始到当前步的所有数据
            current_pop = round_data_norm[: i + 1]

            # 计算当前体积
            current_hv = ind(current_pop)

            # 计算百分比
            hv_percentage = (current_hv / total_hv) * 100

            # 获取对应的 batch_index
            batch_idx = sub_df.iloc[i]["batch_index"]

            hv_records.append({"round_index": r_idx, "batch_index": batch_idx, "hv_percentage": hv_percentage})

    hv_df = pd.DataFrame(hv_records)

    # 4. 绘图
    if hv_df.empty:
        print("No data to plot.")
        return

    plt.figure(figsize=(10, 6))

    # 使用 seaborn 绘制带有置信区间的折线图
    sns.lineplot(
        data=hv_df,
        x="batch_index",
        y="hv_percentage",
        errorbar="sd",  # 显示标准差作为阴影
        linewidth=2.5,
        color="#2ca02c",  # 绿色
        marker="o",  # 增加数据点标记
        markevery=max(1, len(hv_df) // 20),  # 防止点太密
        label="Observed HV",
    )

    plt.title(f"Hypervolume Coverage ({len(target_columns)} Objectives)\nDirections: {direction_tags}", fontsize=14)
    plt.xlabel("Batch Iteration", fontsize=12)
    plt.ylabel("Hypervolume Coverage (%)", fontsize=12)

    # 设置 Y 轴范围，稍微留白
    plt.ylim(0, 105)

    # 添加网格
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right")

    save_path = experiment_dir / "plot_2_hypervolume_coverage.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot 2 saved: {save_path}")


def plot_final_distribution_boxplot(df, target_columns, direction_tags, range_tags, experiment_dir):
    """
    绘制美化版箱型图：所有 column_name_list 最后优化到的最佳值分布。
    """
    # --- 0. 全局风格设置 ---
    sns.set_theme(style="whitegrid", context="talk", font_scale=0.9)
    # 自定义配色：主色调（箱子）和 点色调
    box_color = "#4C72B0"  # 深一点的蓝色，显得专业
    point_color = "#C44E52"  # 柔和的红色

    # 1. 提取每一轮(round)的最终最佳值
    final_best_records = []
    dir_map = dict(zip(target_columns, direction_tags))
    range_map = dict(zip(target_columns, range_tags))

    for r_idx in df["round_index"].unique():
        sub_df = df[df["round_index"] == r_idx]
        record = {"round_index": r_idx}
        for col in target_columns:
            direction = dir_map[col]
            if direction == "max":
                best_val = sub_df[col].max()
            elif direction == "min":
                best_val = sub_df[col].min()
            else:
                raise Exception(f"Unknown direction: {direction}")
            record[col] = best_val
        final_best_records.append(record)

    best_df = pd.DataFrame(final_best_records)

    # 2. 转换数据格式
    melted_df = best_df.melt(id_vars=["round_index"], value_vars=target_columns, var_name="Target", value_name="Best Value")

    # 3. 绘图初始化
    # height 和 aspect 稍微调整，让图看起来更舒展
    g = sns.FacetGrid(melted_df, col="Target", sharey=False, height=5, aspect=0.7, col_wrap=min(len(target_columns), 4))

    # --- 4. 绘图核心优化 ---

    # A. 绘制箱型图
    # boxprops: 设置箱子的透明度(alpha)和边缘线
    # fliersize=0: 隐藏箱型图自带的离群点，因为我们要用 stripplot 画所有点
    # width: 箱子宽度适中
    g.map_dataframe(
        sns.boxplot, y="Best Value", width=0.4, color=box_color, fliersize=0, linewidth=1.5, boxprops=dict(alpha=0.4, edgecolor=box_color)
    )

    # B. 绘制散点图 (数据点)
    # 使用 stripplot 叠加真实数据点
    # edgecolors="white", linewidth=1: 给点加上白色描边，防止背景深色时看不清
    g.map_dataframe(sns.stripplot, y="Best Value", color=point_color, size=6, jitter=0.2, alpha=0.9, edgecolor="white", linewidth=1)

    # 5. 针对每个子图的精细调整
    axes = g.axes.flatten()
    ordered_targets = list(g.col_names)

    for ax, target_name in zip(axes, ordered_targets):
        y_range = range_map.get(target_name)
        direction = dir_map.get(target_name)

        # 计算当前数据的统计值，用于标注（可选）
        current_data = melted_df[melted_df["Target"] == target_name]["Best Value"]
        mean_val = current_data.mean()

        # 绘制一条均值虚线（增强信息量）
        ax.axhline(mean_val, color=box_color, linestyle="--", linewidth=1, alpha=0.6, label=f"Mean: {mean_val:.2f}")

        # 设置 Y 轴范围
        if y_range:
            margin = (y_range[1] - y_range[0]) * 0.1
            ax.set_ylim(y_range[0] - margin, y_range[1] + margin)

        # 标题美化
        # 使用不同的颜色或箭头符号来指示方向
        arrow = "▲" if direction == "max" else "▼"
        color_title = "#2ca02c" if direction == "max" else "#d62728"  # 绿色代表最大化，红色代表最小化（仅作区分，可改）
        # 这里为了保持商务风，统一用黑色，但在文字上区分
        title_text = f"{target_name}\n({arrow} {direction.capitalize()})"
        ax.set_title(title_text, fontsize=12, fontweight="bold", pad=15)

        # 坐标轴美化
        ax.set_ylabel("")  # 移除默认的label，最后统一加或者保持简洁
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))  # 限制Y轴刻度数量，避免拥挤
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)  # 虚线网格

    # 6. 全局布局调整
    # 增加左侧 Y 轴标签
    # 技巧：在 Figure 对象上添加一个 text 作为全局 Y label
    g.fig.text(0.01, 0.5, "Best Optimization Value", va="center", rotation="vertical", fontsize=12, color="gray")

    plt.subplots_adjust(top=0.82, wspace=0.3, left=0.08)  # 留出顶部给大标题，增加子图间距

    # 主标题
    g.fig.suptitle("Distribution of Final Optimization Results", fontsize=16, fontweight="bold", y=0.92, color="#333333")

    # 副标题/说明
    g.fig.text(0.5, 0.88, f"Across {len(df['round_index'].unique())} Independent Rounds", ha="center", fontsize=11, color="gray")

    save_path = experiment_dir / "plot_3_best_value_distribution.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")  # bbox_inches='tight' 防止文字被截断
    plt.close()
    print(f"Plot 3 saved (Beautified): {save_path}")


def plot_optimization_process_scatter(df, target_columns, direction_tags, range_tags, full_space_file, experiment_dir):
    """
    绘制优化过程散点图 (只显示非支配解)：
    1. 背景：利用 full_space_file 计算全空间真实帕累托前沿，并按顺序连成一条浅蓝色线条。
    2. 前景：仅绘制实验结果中找到的非支配解 (Empirical Pareto Front)，以散点显示，颜色根据 Batch Index 渐变。
    """
    df = df[df['round_index'] == 0]
    
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
