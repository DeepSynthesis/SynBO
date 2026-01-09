import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


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
                new_min = y_min if y_min is not None else current_ylim[0]
                new_max = y_max if y_max is not None else current_ylim[1]
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
    绘制箱型图：所有 column_name_list 最后优化到的最佳值（最大或最小）的分布。
    根据 direction_tags 决定取最大还是最小。
    根据 range_tags 设定 Y 轴范围。
    """
    # 1. 提取每一轮(round)的最终最佳值
    final_best_records = []
    # 建立映射方便后续查找
    dir_map = dict(zip(target_columns, direction_tags))
    range_map = dict(zip(target_columns, range_tags))
    for r_idx in df["round_index"].unique():
        sub_df = df[df["round_index"] == r_idx]
        record = {"round_index": r_idx}

        for col in target_columns:
            direction = dir_map[col]
            if direction == "max":
                # 越大越好，取最大值
                best_val = sub_df[col].max()
            elif direction == "min":
                # 越小越好，取最小值
                best_val = sub_df[col].min()
            else:
                raise Exception(f"{direction}???")

            record[col] = best_val

        final_best_records.append(record)
    best_df = pd.DataFrame(final_best_records)
    # 2. 转换数据格式为 Long Format
    melted_df = best_df.melt(id_vars=["round_index"], value_vars=target_columns, var_name="Target", value_name="Best Value")
    # 3. 绘图
    # sharey=False 非常重要，因为不同目标的范围不同
    g = sns.FacetGrid(melted_df, col="Target", sharey=False, height=5, aspect=0.8, col_wrap=min(len(target_columns), 4))

    # 绘制箱型图和散点图
    g.map_dataframe(sns.boxplot, y="Best Value", width=0.5, color="skyblue")
    g.map_dataframe(sns.stripplot, y="Best Value", color="red", size=5, jitter=True, alpha=0.7)
    # 4. 针对每个子图设置特定的 Y 轴范围和标题
    # g.axes 是一个 numpy array，包含所有的子图对象
    axes = g.axes.flatten()

    # 获取 FacetGrid 每一列对应的 Target 名称顺序
    # 注意：Seaborn 默认按字母顺序或出现的顺序排列，这里通常和 col_order 一致
    # 为了保险，直接遍历 g.col_names (Seaborn 版本差异可能导致属性名不同，通常 g.col_names 是列名列表)
    # 如果 g.col_names 不可用，使用 melted_df['Target'].unique() 但需确保顺序一致

    ordered_targets = list(g.col_names)
    for ax, target_name in zip(axes, ordered_targets):
        # 获取该 Target 对应的范围和方向
        y_range = range_map.get(target_name)
        direction = dir_map.get(target_name)

        # 设置 Y 轴范围
        if y_range:
            # y_range 是 tuple (min, max)
            # 稍微留一点余地防止点压在边框上，或者直接严格限制
            ax.set_ylim(y_range[0], y_range[1])

        # 更新标题，加上方向提示
        title_suffix = " (Maximize)" if direction == "max" else " (Minimize)"
        ax.set_title(f"{target_name}{title_suffix}", fontsize=11)

        # 设置 Y 轴标签
        ax.set_ylabel("Best Value Found")
    # 加上全局标题
    plt.subplots_adjust(top=0.85)
    g.fig.suptitle("Distribution of Best Found Values across Rounds", fontsize=14)
    save_path = experiment_dir / "plot_3_best_value_distribution.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Plot 3 saved: {save_path}")


def plot_optimization_process_scatter(df, target_columns, direction_tags, range_tags, full_space_file, experiment_dir):
    """
    绘制优化过程散点图：
    1. 背景：利用 full_space_file 计算并绘制真实的帕累托前沿 (Light Blue)。
    2. 前景：绘制实验过程中的点，颜色根据 Batch Index 渐变 (Blues)，越靠后越深。
    支持 2D (2目标) 和 3D (3目标) 绘图。超过3目标将仅绘制前两维。
    Args:
        df: 实验结果 DataFrame。
        target_columns: 目标列名列表。
        direction_tags: 优化方向列表 ['min', 'max', ...]。
        range_tags: 坐标轴范围列表 [(min, max), ...]。
        full_space_file: 全空间数据文件路径。
        experiment_dir: 保存路径。
    """
    try:
        from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    except ImportError:
        print("Error: 'pymoo' library is required for Pareto calculation. Run 'pip install pymoo'.")
        return
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    n_targets = len(target_columns)

    # 1. 读取全空间数据
    if str(full_space_file).endswith(".csv"):
        full_df = pd.read_csv(full_space_file)
    else:
        full_df = pd.read_excel(full_space_file)
    full_data_raw = full_df[target_columns].values
    # 2. 计算全空间数据的帕累托前沿 (True Pareto Front)
    # Pymoo 默认最小化，因此需要根据 direction_tags 对 max 目标取反
    sorting_data = full_data_raw.copy()
    for i, direction in enumerate(direction_tags):
        if direction == "max":
            sorting_data[:, i] *= -1  # 变为最小化问题进行排序

    # 执行非支配排序
    nds = NonDominatedSorting()
    fronts = nds.do(sorting_data)

    # 取出第一层前沿 (Rank 0) 的索引
    pf_indices = fronts[0]
    true_pf_data = full_data_raw[pf_indices]  # 注意：绘图要用原始数据，不是取反后的
    # 3. 准备实验数据
    exp_data = df[target_columns].values
    batch_indices = df["batch_index"].values
    # 4. 开始绘图
    # 根据目标数量选择绘图模式
    if n_targets == 2:
        _plot_2d_scatter(true_pf_data, exp_data, batch_indices, target_columns, range_tags, direction_tags, experiment_dir)
    elif n_targets == 3:
        _plot_3d_scatter(true_pf_data, exp_data, batch_indices, target_columns, range_tags, direction_tags, experiment_dir)
    else:
        print(f"Warning: {n_targets} objectives detected. Plotting only the first 2 dimensions for visualization.")
        _plot_2d_scatter(true_pf_data, exp_data, batch_indices, target_columns[:2], range_tags[:2], direction_tags[:2], experiment_dir)


def _plot_2d_scatter(pf_data, exp_data, batch_indices, targets, ranges, directions, save_dir):
    """内部辅助函数：绘制 2D 图"""
    plt.figure(figsize=(8, 8))

    # Layer 1: True Pareto Front (背景)
    plt.scatter(
        pf_data[:, 0], pf_data[:, 1], c="skyblue", alpha=0.4, s=30, label="True Pareto Front", zorder=0  # 浅蓝色  # 半透明  # 放在最底层
    )
    # Layer 2: Optimization Process (前景)
    # 使用 Blues colormap，c 指定为 batch_indices 实现渐变
    sc = plt.scatter(
        exp_data[:, 0],
        exp_data[:, 1],
        c=batch_indices,
        cmap="Blues",
        edgecolors="k",  # 加黑色边框以区分浅色背景
        linewidth=0.5,
        s=50,
        alpha=0.9,
        label="Observed Samples",
        zorder=10,  # 放在上层
    )
    # 添加 Colorbar
    cbar = plt.colorbar(sc)
    cbar.set_label("Batch Iteration")
    # 设置坐标轴范围 (如果提供了 range_tags)
    if ranges:
        plt.xlim(ranges[0])
        plt.ylim(ranges[1])
    # 标签与标题
    plt.xlabel(f"{targets[0]} ({directions[0]})")
    plt.ylabel(f"{targets[1]} ({directions[1]})")
    plt.title("Optimization Process & Pareto Front Coverage")

    # Legend (只显示 True PF，因为 Scatter 已经有 Colorbar)
    plt.legend(loc="best")
    plt.grid(True, linestyle="--", alpha=0.5)
    save_path = save_dir / "plot_4_optimization_process_2d.png"
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
