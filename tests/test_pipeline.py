from datetime import datetime
import os
from pathlib import Path
import shutil
from matplotlib import pyplot as plt
import pandas as pd
from rxnopt import ReactionOptimizer
from rxnopt.descriptor.spoc_desc import calc_spoc_desc
from rxnopt.utils import load_desc_dict, get_prev_rxn

import seaborn as sns
import numpy as np


def compute_hypervolume_2d(points, ref_point):
    """
    计算2D情况下的超体积 (Hypervolume)

    Args:
        points: 目标点数组，形状为 (n_points, 2)
        ref_point: 参考点，形状为 (2,)

    Returns:
        float: 超体积值
    """
    points = np.array(points)
    ref_point = np.array(ref_point)

    # 过滤掉被参考点支配的点
    valid_mask = np.all(points > ref_point, axis=1)
    if not np.any(valid_mask):
        return 0.0

    valid_points = points[valid_mask]

    # 计算Pareto前沿
    pareto_front = get_pareto_front(valid_points)

    if len(pareto_front) == 0:
        return 0.0

    # 对Pareto前沿按第一个目标排序
    pareto_front = pareto_front[np.argsort(pareto_front[:, 0])]

    # 计算超体积
    hv = 0.0
    for i in range(len(pareto_front)):
        if i == 0:
            # 第一个点
            width = pareto_front[i, 0] - ref_point[0]
            height = pareto_front[i, 1] - ref_point[1]
        else:
            # 后续点
            width = pareto_front[i, 0] - pareto_front[i - 1, 0]
            height = pareto_front[i, 1] - ref_point[1]

        hv += width * height

    return hv


def get_pareto_front(points):
    """
    获取Pareto前沿点

    Args:
        points: 目标点数组，形状为 (n_points, n_objectives)

    Returns:
        np.array: Pareto前沿点
    """
    points = np.array(points)
    n_points = len(points)

    if n_points == 0:
        return np.array([])

    # 对于每个点，检查是否被其他点支配
    is_pareto = np.ones(n_points, dtype=bool)

    for i in range(n_points):
        for j in range(n_points):
            if i != j:
                # 检查点j是否支配点i
                if np.all(points[j] >= points[i]) and np.any(points[j] > points[i]):
                    is_pareto[i] = False
                    break

    return points[is_pareto]


def fill_done_dir(i, date):
    current_df = pd.read_csv(f"results/batch-{i}_{date}.csv")
    current_df.drop(columns=["yield", "cost"], inplace=True)
    HTE_df = pd.read_csv(f"dataset/B-H_dataset.csv")
    merged_df = pd.merge(
        current_df,
        HTE_df[["base", "ligand", "solvent", "concentration", "temperature", "yield", "cost"]],
        on=["base", "ligand", "solvent", "concentration", "temperature"],
        how="left",
    )
    merged_df.to_csv(f"results/batch-{i}_{date}.csv", index=False)


date = datetime.now().strftime("%Y%m%d")
for f in Path("results/").glob(f"batch-*.csv"):
    os.remove(f)


# def generate_onehot():
#     for name in ["base", "ligand", "solvent"]:
#         df = pd.read_csv(f"dataset/descriptors/{name}_dft.csv")
#         calc_spoc_desc(df[f"{name}_file_name"], save_path=f"dataset/descriptors/{name}.csv", fp_type="RDKit", desc_type_to_filename=True)


# generate_onehot()
# exit()
reagent_types = ["base", "ligand", "solvent", "concentration", "temperature"]
index_col = [f"{r}_file_name" for r in reagent_types]
name_suffix = ["_dft", "_dft", "_dft", None, None]
opt_direct_info = [{"opt_direct": "max", "opt_range": [0, 100]}, {"opt_direct": "min", "opt_range": [0.02, 0.5]}]  # cost(min), yield(max)

desc_dict, condition_dict = load_desc_dict(
    reagent_types=reagent_types, desc_dir="dataset/descriptors", name_suffix=name_suffix, return_condition_dict=True, index_col=index_col
)

for i in range(10):
    rxn_opt = ReactionOptimizer(opt_metrics=["yield", "cost"], opt_direct_info=opt_direct_info, opt_type="auto")
    rxn_opt.load_rxn_space(condition_dict=condition_dict)
    rxn_opt.load_desc(desc_dict=desc_dict)
    if i > 0:
        rxn_opt.load_prev_rxn(prev_rxn_info=get_prev_rxn(file_pattern=f"results/batch-*.csv"))
    if i == 0:
        rxn_opt.initialize(batch_size=5, desc_normalize="zscore", sampling_method="cvt", refine_desc="pass")
    else:
        rxn_opt.optimize(batch_size=5, desc_normalize="zscore", mc_num_samples=32, max_batch_size=32, refine_desc="pass")
    rxn_opt.save_results(save_dir="results")

    fill_done_dir(i, date)


def plot_optimization_process(file_pattern, save_path="results/optimization_process.png"):
    """
    绘制优化过程中每个目标值随batch_id变化的曲线
    使用散点图+箱型图组合展示
    """
    prev_rxn_df = get_prev_rxn(file_pattern=file_pattern)
    target_columns = ["yield", "cost"]
    sns.set_style("whitegrid")
    plt.figure(figsize=(15, 6))

    for i, target in enumerate(target_columns, 1):
        plt.subplot(1, len(target_columns), i)
        sns.boxplot(data=prev_rxn_df, x="batch", y=target, color="lightblue")
        sns.stripplot(data=prev_rxn_df, x="batch", y=target, size=6, alpha=0.8, jitter=True, color="red")
        plt.title(f"{target.capitalize()} vs Batch ID", fontsize=14, fontweight="bold")
        plt.xlabel("Batch ID", fontsize=12)
        plt.ylabel(target.capitalize(), fontsize=12)

        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


plot_optimization_process(file_pattern=f"results/batch-*.csv")


def calculate_max_hv_from_dataset(dataset_path="dataset/B-H_dataset.csv", opt_direct_info=None):
    """
    从完整数据集计算最大超体积
    """
    if opt_direct_info is None:
        opt_direct_info = [{"opt_direct": "min", "opt_range": [0, 0.5]}, {"opt_direct": "max", "opt_range": [0, 100]}]

    dataset_df = pd.read_csv(dataset_path)

    # 提取目标值，按照opt_metrics的顺序: ["cost", "yield"]
    objectives = dataset_df[["cost", "yield"]].values.copy()

    # 根据opt_direct_info调整目标方向
    for i, direction_info in enumerate(opt_direct_info):
        if direction_info["opt_direct"] == "min":
            objectives[:, i] = -objectives[:, i]  # 最小化目标取负值转为最大化

    # 对于超体积计算，参考点必须被所有点严格支配
    # 即参考点的每个维度都要小于所有目标点的对应维度
    min_vals = objectives.min(axis=0)
    max_vals = objectives.max(axis=0)

    # 设置参考点为最小值再减去一个合理的偏移量
    range_vals = max_vals - min_vals
    ref_point = min_vals - np.maximum(range_vals * 0.1, 1.0)

    print(
        f"Original objectives range: cost [{dataset_df['cost'].min():.6f}, {dataset_df['cost'].max():.6f}], yield [{dataset_df['yield'].min():.6f}, {dataset_df['yield'].max():.6f}]"
    )
    print(f"Transformed objectives range: {objectives.min(axis=0)} to {objectives.max(axis=0)}")
    print(f"Reference point: {ref_point}")

    # 使用自定义的超体积计算
    max_hv = compute_hypervolume_2d(objectives, ref_point)

    print(f"Calculated max HV: {max_hv}")

    return max_hv, ref_point


def plot_hv_percentage(file_pattern, dataset_path="dataset/B-H_dataset.csv", opt_direct_info=None, save_path="results/hv_percentage.png"):
    """
    绘制HV百分比随batch变化的图
    """
    if opt_direct_info is None:
        opt_direct_info = [{"opt_direct": "min", "opt_range": [0, 0.5]}, {"opt_direct": "max", "opt_range": [0, 100]}]

    # 计算全空间最大HV和参考点
    max_hv, ref_point = calculate_max_hv_from_dataset(dataset_path, opt_direct_info)

    if max_hv <= 0:
        print("Warning: max_hv is 0 or negative, cannot calculate meaningful percentages")
        return

    # 获取优化过程数据
    prev_rxn_df = get_prev_rxn(file_pattern=file_pattern)

    # 计算每个batch的当前最大HV
    batch_hv_percentages = []
    batches = sorted(prev_rxn_df["batch"].unique())

    for batch in batches:
        # 获取到当前batch为止的所有数据
        current_data = prev_rxn_df[prev_rxn_df["batch"] <= batch]

        # 提取目标值，按照opt_metrics的顺序: ["cost", "yield"]
        objectives = current_data[["cost", "yield"]].values.copy()

        # 根据opt_direct_info调整目标方向
        for i, direction_info in enumerate(opt_direct_info):
            if direction_info["opt_direct"] == "min":
                objectives[:, i] = -objectives[:, i]  # 最小化目标取负值转为最大化

        # 使用自定义的超体积计算
        current_hv = compute_hypervolume_2d(objectives, ref_point)
        hv_percentage = (current_hv / max_hv) * 100

        print(f"Batch {batch}: current_hv={current_hv:.6f}, percentage={hv_percentage:.2f}%")

        batch_hv_percentages.append(hv_percentage)

    # 绘图
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))

    plt.plot(batches, batch_hv_percentages, marker="o", linewidth=2, markersize=8, color="darkgreen")
    plt.title("Hypervolume Percentage vs Batch ID", fontsize=14, fontweight="bold")
    plt.xlabel("Batch ID", fontsize=12)
    plt.ylabel("HV Percentage (%)", fontsize=12)
    plt.grid(True, alpha=0.3)

    # 添加最终百分比标注
    if batch_hv_percentages:
        final_percentage = batch_hv_percentages[-1]
        plt.annotate(
            f"Final: {final_percentage:.2f}%",
            xy=(batches[-1], final_percentage),
            xytext=(batches[-1] - 1, final_percentage + 5),
            arrowprops=dict(arrowstyle="->", color="red"),
            fontsize=10,
            fontweight="bold",
            color="red",
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"HV percentage plot saved to {save_path}")


plot_hv_percentage(file_pattern=f"results/batch-*.csv", opt_direct_info=opt_direct_info)
