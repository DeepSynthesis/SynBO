#!/usr/bin/env python3
"""
Reagents Reduction Statistic Analysis

This script performs analysis on prohibited reagents from benchmark results:
1. Reads prohibited_reagent_*.json files and counts reagent frequencies by category
2. Reads B-H_HTE.csv dataset and calculates yield and cost statistics for each reagent
3. Creates dual-axis plots showing both frequency and yield/cost distributions
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from collections import defaultdict
from pathlib import Path


def read_prohibited_reagents(json_path_pattern):
    """
    Read all prohibited_reagent_*.json files and count frequencies.
    
    Returns:
        dict: {category: {reagent_name: frequency}}
    """
    # 存储所有类别中的试剂频率
    category_reagent_freq = defaultdict(lambda: defaultdict(int))
    
    # 找到所有匹配的JSON文件
    json_files = sorted(glob.glob(json_path_pattern))
    print(f"Found {len(json_files)} prohibited reagent files")
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 统计每个类别的试剂出现次数
        for category, reagents in data.items():
            if isinstance(reagents, list):
                for reagent in reagents:
                    category_reagent_freq[category][reagent] += 1
            else:
                # 处理可能的单值情况
                category_reagent_freq[category][reagents] += 1
    
    return category_reagent_freq


def read_dataset_yield_cost(csv_path):
    """
    Read B-H_HTE.csv and calculate yield*(0.42-cost) statistics for each reagent.
    
    Returns:
        dict: {category: {reagent_name: {'values': []}}}
    """
    df = pd.read_csv(csv_path)
    
    # 需要分析的列（类别）
    category_columns = ['base', 'ligand', 'solvent', 'concentration', 'temperature']
    
    # 存储每个类别的试剂数据
    category_reagent_data = defaultdict(lambda: defaultdict(lambda: {'values': []}))
    
    for col in category_columns:
        if col not in df.columns:
            continue
            
        # 按该列分组
        grouped = df.groupby(col)
        
        for reagent_name, group in grouped:
            # 计算 yield * (0.42 - cost)
            values = (group['yield'] * (0.42 - group['cost'])).tolist()
            
            category_reagent_data[col][reagent_name]['values'] = values
    
    return category_reagent_data


def create_dual_axis_plot(category, reagent_freq, reagent_data, output_path):
    """
    Create a dual-axis plot for a category:
    - Left axis: Frequency line plot
    - Right axis: Scatter plot of yield*(0.42-cost) values with mean and std annotations
    
    Args:
        category: Category name (e.g., 'base', 'ligand')
        reagent_freq: {reagent_name: frequency}
        reagent_data: {reagent_name: {'values': []}}
        output_path: Path to save the figure
    """
    # 获取所有试剂名（从frequency和data的并集）
    freq_reagents = set(reagent_freq.keys())
    data_reagents = set(reagent_data.keys())
    all_reagents = list(freq_reagents | data_reagents)
    
    if len(all_reagents) == 0:
        print(f"  Warning: No data for category '{category}'")
        return
    
    # 计算每个试剂的平均值，用于排序
    reagent_stats = {}
    for reagent in all_reagents:
        if reagent in reagent_data:
            values = reagent_data[reagent]['values']
            mean_val = np.mean(values) if values else 0
            std_val = np.std(values) if values else 0
            reagent_stats[reagent] = {'mean': mean_val, 'std': std_val, 'count': len(values)}
        else:
            reagent_stats[reagent] = {'mean': 0, 'std': 0, 'count': 0}
    
    # 按平均值从高到低排序
    all_reagents = sorted(all_reagents, key=lambda r: reagent_stats[r]['mean'], reverse=True)
    
    # 准备数据
    frequencies = [reagent_freq.get(r, 0) for r in all_reagents]
    
    # 准备散点图数据和统计信息
    scatter_data = []  # [(position, value), ...]
    mean_values = []   # 每个试剂的平均值
    std_values = []    # 每个试剂的标准差
    
    for i, reagent in enumerate(all_reagents):
        if reagent in reagent_data:
            values = reagent_data[reagent]['values']
            for v in values:
                scatter_data.append((i, v))
            mean_values.append(reagent_stats[reagent]['mean'])
            std_values.append(reagent_stats[reagent]['std'])
        else:
            mean_values.append(0)
            std_values.append(0)
    
    # 创建图
    fig, ax1 = plt.subplots(figsize=(max(18, len(all_reagents) * 1.5), 10))
    
    # 左轴：频率折线图
    color_freq = '#2E86AB'  # 蓝色
    ax1.plot(range(len(all_reagents)), frequencies, 'o-', color=color_freq, 
             linewidth=3, markersize=10, label='Frequency', zorder=3)
    ax1.set_xlabel('Reagent', fontsize=18)
    ax1.set_ylabel('Frequency', color=color_freq, fontsize=18)
    ax1.tick_params(axis='y', labelcolor=color_freq, labelsize=16)
    ax1.tick_params(axis='x', labelsize=16)
    ax1.set_xticks(range(len(all_reagents)))
    ax1.set_xticklabels(all_reagents, rotation=45, ha='right', fontsize=16)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xlim(-0.5, len(all_reagents) - 0.5)
    
    # 右轴：散点图
    ax2 = ax1.twinx()
    
    if scatter_data:
        positions = [p for p, v in scatter_data]
        values = [v for p, v in scatter_data]
        # 添加一些随机抖动到x位置，避免点重叠
        jittered_positions = [p + np.random.uniform(-0.2, 0.2) for p in positions]
        ax2.scatter(jittered_positions, values, alpha=0.3, s=25, 
                   color='#A23B72', label='HV', zorder=2)
    
    # 绘制平均值线
    ax2.plot(range(len(all_reagents)), mean_values, 's-', color='#E63946', 
             linewidth=3, markersize=10, label='Mean', zorder=4, alpha=0.8)
    
    # 绘制标准差范围（均值±标准差）
    mean_plus_std = [m + s for m, s in zip(mean_values, std_values)]
    mean_minus_std = [max(0, m - s) for m, s in zip(mean_values, std_values)]
    ax2.fill_between(range(len(all_reagents)), mean_minus_std, mean_plus_std, 
                     alpha=0.2, color='#E63946', label='±1 Std Dev', zorder=1)
    
    ax2.set_ylabel('HV', color='#555555', fontsize=18)
    ax2.tick_params(axis='y', labelcolor='#555555', labelsize=16)
    
    # 重新设置x轴标签
    ax1.set_xticks(range(len(all_reagents)))
    ax1.set_xticklabels(all_reagents, rotation=45, ha='right', fontsize=16)
    
    # 添加图例
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Line2D([0], [0], color=color_freq, linewidth=3, marker='o', label='Frequency'),
        Line2D([0], [0], color='#A23B72', marker='o', linestyle='None', 
               markersize=8, label='HV'),
        Line2D([0], [0], color='#E63946', linewidth=3, marker='s', label='Mean'),
        Patch(facecolor='#E63946', alpha=0.2, label='±1 Std Dev')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=16)
    
    # 删除标题
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved plot: {output_path}")


def print_statistics(category, reagent_freq, reagent_data):
    """打印统计信息"""
    print(f"\n{'='*60}")
    print(f"Category: {category}")
    print(f"{'='*60}")
    
    all_reagents = sorted(set(reagent_freq.keys()) | set(reagent_data.keys()))
    
    for reagent in all_reagents:
        freq = reagent_freq.get(reagent, 0)
        
        if reagent in reagent_data:
            values = reagent_data[reagent]['values']
            
            value_mean = np.mean(values) if values else 0
            value_std = np.std(values) if values else 0
            
            print(f"  {reagent}:")
            print(f"    Frequency: {freq}")
            print(f"    Yield×(0.42-Cost): {value_mean:.4f} ± {value_std:.4f} (n={len(values)})")
        else:
            print(f"  {reagent}:")
            print(f"    Frequency: {freq}")
            print(f"    Yield×(0.42-Cost): No data in dataset")


def main():
    # 路径配置
    json_path_pattern = "benchmark/results/multiple_20260327_094215/prohibited_reagent_*.json"
    csv_path = "benchmark/datasets/HTE_datasets/B-H_HTE/B-H_HTE.csv"
    output_dir = "benchmark/results/multiple_20260327_094215/analysis"
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("Step 1: Reading prohibited reagents from JSON files...")
    category_reagent_freq = read_prohibited_reagents(json_path_pattern)
    print(f"  Found categories: {list(category_reagent_freq.keys())}")
    for cat, reagents in category_reagent_freq.items():
        print(f"    {cat}: {len(reagents)} unique reagents")
    
    print("\nStep 2: Reading dataset and calculating yield/cost statistics...")
    category_reagent_data = read_dataset_yield_cost(csv_path)
    print(f"  Analyzed categories: {list(category_reagent_data.keys())}")
    for cat, reagents in category_reagent_data.items():
        print(f"    {cat}: {len(reagents)} unique reagents")
    
    print("\nStep 3: Generating plots and statistics...")
    # 合并所有类别
    all_categories = set(category_reagent_freq.keys()) | set(category_reagent_data.keys())
    
    for category in sorted(all_categories):
        reagent_freq = dict(category_reagent_freq.get(category, {}))
        reagent_data = dict(category_reagent_data.get(category, {}))
        
        # 打印统计信息
        print_statistics(category, reagent_freq, reagent_data)
        
        # 创建图表
        output_path = f"{output_dir}/{category}_frequency_yield_cost.png"
        create_dual_axis_plot(category, reagent_freq, reagent_data, output_path)
    
    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"Plots saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()