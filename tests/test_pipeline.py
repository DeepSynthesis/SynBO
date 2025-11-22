from datetime import datetime
import os
from pathlib import Path
import shutil
from matplotlib import pyplot as plt
import pandas as pd
from rxnopt import ReactionOptimizer
from rxnopt.utils import load_desc_dict, get_prev_rxn

import seaborn as sns
import numpy as np
from pymoo.indicators.hv import HV


def fill_done_dir(i, date):
    current_df = pd.read_csv(f"results/batch-{i}_{date}.csv")
    current_df.drop(columns=["cost", "yield"], inplace=True)
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

reagent_types = ["base", "ligand", "solvent", "concentration", "temperature"]
index_col = [f"{r}_file_name" for r in reagent_types]
name_suffix = ["_dft", "_dft", "_dft", None, None]
opt_direct_info = [{"opt_direct": "min", "opt_range": [0, 0.5]}, {"opt_direct": "max", "opt_range": [0, 100]}]  # cost(min), yield(max)

desc_dict, condition_dict = load_desc_dict(
    reagent_types=reagent_types, desc_dir="dataset/descriptors", name_suffix=name_suffix, return_condition_dict=True, index_col=index_col
)

for i in range(10):
    rxn_opt = ReactionOptimizer(opt_metrics=["cost", "yield"], opt_direct_info=opt_direct_info, opt_type="auto")
    rxn_opt.load_rxn_space(condition_dict=condition_dict)
    rxn_opt.load_desc(desc_dict=desc_dict)
    if i > 0:
        rxn_opt.load_prev_rxn(prev_rxn_info=get_prev_rxn(file_pattern=f"results/batch-*.csv"))
    rxn_opt.run()
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
    
    # 定义参考点 (对于最大化问题，参考点应该在所有目标值之下)
    ref_point = np.array([objectives.min(axis=0)[j] - 1.0 for j in range(objectives.shape[1])])
    
    # 计算超体积
    hv_indicator = HV(ref_point=ref_point)
    max_hv = hv_indicator(objectives)
    
    return max_hv, ref_point


def plot_hv_percentage(file_pattern, dataset_path="dataset/B-H_dataset.csv", opt_direct_info=None, save_path="results/hv_percentage.png"):
    """
    绘制HV百分比随batch变化的图
    """
    if opt_direct_info is None:
        opt_direct_info = [{"opt_direct": "min", "opt_range": [0, 0.5]}, {"opt_direct": "max", "opt_range": [0, 100]}]
    
    # 计算全空间最大HV和参考点
    max_hv, ref_point = calculate_max_hv_from_dataset(dataset_path, opt_direct_info)
    
    # 获取优化过程数据
    prev_rxn_df = get_prev_rxn(file_pattern=file_pattern)
    
    # 计算每个batch的当前最大HV
    batch_hv_percentages = []
    batches = sorted(prev_rxn_df["batch"].unique())
    
    hv_indicator = HV(ref_point=ref_point)
    
    for batch in batches:
        # 获取到当前batch为止的所有数据
        current_data = prev_rxn_df[prev_rxn_df["batch"] <= batch]
        
        # 提取目标值，按照opt_metrics的顺序: ["cost", "yield"]
        objectives = current_data[["cost", "yield"]].values.copy()
        
        # 根据opt_direct_info调整目标方向
        for i, direction_info in enumerate(opt_direct_info):
            if direction_info["opt_direct"] == "min":
                objectives[:, i] = -objectives[:, i]  # 最小化目标取负值转为最大化
        
        # 计算当前HV
        current_hv = hv_indicator(objectives)
        hv_percentage = (current_hv / max_hv) * 100
        
        batch_hv_percentages.append(hv_percentage)
    
    # 绘图
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    
    plt.plot(batches, batch_hv_percentages, marker='o', linewidth=2, markersize=8, color='darkgreen')
    plt.title("Hypervolume Percentage vs Batch ID", fontsize=14, fontweight="bold")
    plt.xlabel("Batch ID", fontsize=12)
    plt.ylabel("HV Percentage (%)", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 添加最终百分比标注
    final_percentage = batch_hv_percentages[-1]
    plt.annotate(f'Final: {final_percentage:.2f}%', 
                xy=(batches[-1], final_percentage), 
                xytext=(batches[-1]-1, final_percentage+5),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, fontweight='bold', color='red')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


plot_hv_percentage(file_pattern=f"results/batch-*.csv", opt_direct_info=opt_direct_info)
