from datetime import datetime
import os
from pathlib import Path
import shutil
from matplotlib import pyplot as plt
import pandas as pd
from rxnopt import ReactionOptimizer
from rxnopt.utils import load_desc_dict, get_prev_rxn

import seaborn as sns


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
opt_direct_info = [{"opt_direct": "max", "opt_range": [0, 100]}]  # ,{"opt_direct": "max", "opt_range": [0, 0.5]}]

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
    
    # 获取目标值列（除了batch列）
    target_columns = ["yield", "cost"]
    
    # 设置图形风格
    sns.set_style("whitegrid")
    plt.figure(figsize=(15, 6))
    
    # 为每个目标值创建子图
    for i, target in enumerate(target_columns, 1):
        plt.subplot(1, len(target_columns), i)
        
        # 创建组合图：箱型图 + 散点图
        # 箱型图显示每个batch的分布
        sns.boxplot(data=prev_rxn_df, x="batch", y=target, alpha=0.6, color="lightblue")
        
        # 散点图显示具体数据点
        sns.stripplot(data=prev_rxn_df, x="batch", y=target, 
                     size=6, alpha=0.8, jitter=True, color="red")
        
        # 设置标题和标签
        plt.title(f"{target.capitalize()} vs Batch ID", fontsize=14, fontweight='bold')
        plt.xlabel("Batch ID", fontsize=12)
        plt.ylabel(target.capitalize(), fontsize=12)
        
        # 旋转x轴标签以避免重叠
        plt.xticks(rotation=45)
        
        # 添加网格
        plt.grid(True, alpha=0.3)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# 调用绘图函数
plot_optimization_process(file_pattern=f"results/batch-*.csv")
