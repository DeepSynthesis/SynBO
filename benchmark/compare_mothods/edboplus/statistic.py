import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def get_all_reagents_from_descriptors(descriptor_dir: Path) -> dict:
    """
    从描述符文件中读取所有可用的试剂名称

    Returns:
        dict: 包含 'ligand', 'base', 'solvent' 三个键的字典，值为试剂名称列表
    """
    reagents = {"ligand": [], "base": [], "solvent": []}

    # 读取ligand描述符 (第一列index是ligand名称)
    ligand_file = descriptor_dir / "ligand_dft.csv"
    if ligand_file.exists():
        df = pd.read_csv(ligand_file, index_col=0)
        reagents["ligand"] = list(df.index)

    # 读取base描述符 (第一列index是base名称)
    base_file = descriptor_dir / "base_dft.csv"
    if base_file.exists():
        df = pd.read_csv(base_file, index_col=0)
        reagents["base"] = list(df.index)

    # 读取solvent描述符 (第一列index是solvent名称)
    solvent_file = descriptor_dir / "solvent_dft.csv"
    if solvent_file.exists():
        df = pd.read_csv(solvent_file, index_col=0)
        reagents["solvent"] = list(df.index)

    return reagents


def count_unused_per_batch(all_reagents: dict, results_dir: Path) -> dict:
    """
    统计每个试剂在每个batch中未使用的次数

    Returns:
        dict: 键为试剂名称，值为该试剂在多少个batch中未被使用
    """
    # 获取所有batch文件
    batch_files = sorted(results_dir.glob("batch_*.csv"))
    total_batches = len(batch_files)

    # 初始化计数器：每个试剂未使用的次数
    unused_counts = {
        "ligand": {reagent: 0 for reagent in all_reagents["ligand"]},
        "base": {reagent: 0 for reagent in all_reagents["base"]},
        "solvent": {reagent: 0 for reagent in all_reagents["solvent"]},
    }

    print(f"\n   共找到 {total_batches} 个batch文件")

    for batch_file in batch_files:
        df = pd.read_csv(batch_file)

        # 检查ligand
        if "ligand" in df.columns:
            used_ligands = set(df["ligand"].dropna().unique())
            for ligand in all_reagents["ligand"]:
                if ligand not in used_ligands:
                    unused_counts["ligand"][ligand] += 1

        # 检查base
        if "base" in df.columns:
            used_bases = set(df["base"].dropna().unique())
            for base in all_reagents["base"]:
                if base not in used_bases:
                    unused_counts["base"][base] += 1

        # 检查solvent
        if "solvent" in df.columns:
            used_solvents = set(df["solvent"].dropna().unique())
            for solvent in all_reagents["solvent"]:
                if solvent not in used_solvents:
                    unused_counts["solvent"][solvent] += 1

    return unused_counts, total_batches


def plot_unused_reagents(unused_counts: dict, total_batches: int, output_dir: Path = None):
    """
    使用seaborn绘制未使用试剂的频数图
    所有试剂画在一张图上，同一类型试剂相同颜色，使用Arial字体
    """
    # 设置Arial字体
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    # 为每种试剂类型分配颜色
    type_colors = {"Ligand": "#1f77b4", "Base": "#ff7f0e", "Solvent": "#2ca02c"}  # 蓝色  # 橙色  # 绿色

    # 准备数据：过滤掉从未未使用的试剂（计数为0的）
    data = []
    for reagent_type, counts in unused_counts.items():
        for reagent, count in counts.items():
            if count > 0:  # 只显示有未使用记录的试剂
                data.append(
                    {
                        "Reagent Type": reagent_type.capitalize(),
                        "Reagent": reagent,
                        "Unused Count": count,
                        "Unused Rate": count / total_batches * 100,
                    }
                )

    if not data:
        print("所有试剂在每个batch中都被使用了！")
        return None

    df = pd.DataFrame(data)

    # 按试剂类型和未使用次数排序，便于展示（Ligand在前，然后是Base，Solvent）
    type_order = {"Ligand": 0, "Base": 1, "Solvent": 2}
    df["TypeOrder"] = df["Reagent Type"].map(type_order)
    df = df.sort_values(["TypeOrder", "Unused Count"], ascending=[True, True])

    # 创建单张图
    fig, ax = plt.subplots(figsize=(16, 10))

    # 为每个试剂绘制条形图，同一类型使用相同颜色
    y_positions = range(len(df))
    bar_colors = [type_colors[rt] for rt in df["Reagent Type"]]
    bars = ax.barh(y_positions, df["Unused Count"], color=bar_colors, edgecolor="black", linewidth=0.5)

    # 设置y轴标签
    ax.set_yticks(y_positions)
    ax.set_yticklabels(df["Reagent"], fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)

    # 设置标题和标签（大字体）
    ax.set_title(f"Unused Reagents Frequency (out of {total_batches} batches)", fontsize=20, fontweight="bold", pad=20)
    ax.set_xlabel(f"Number of Batches Not Used", fontsize=22)
    ax.set_ylabel("Reagent", fontsize=18)

    # 添加数值标签
    for i, (idx_row, row) in enumerate(df.iterrows()):
        ax.text(
            row["Unused Count"] + 0.15,
            i,
            f"",
            color="black",
            fontsize=18,
            va="center",
            fontweight="bold",
        )

    # 添加网格线
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    # 添加图例说明试剂类型（使用实际的颜色）
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=type_colors["Ligand"], edgecolor="black", label=f'Ligand ({len(df[df["Reagent Type"]=="Ligand"])})'),
        Patch(facecolor=type_colors["Base"], edgecolor="black", label=f'Base ({len(df[df["Reagent Type"]=="Base"])})'),
        Patch(facecolor=type_colors["Solvent"], edgecolor="black", label=f'Solvent ({len(df[df["Reagent Type"]=="Solvent"])})'),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=20, title="Reagent Types", title_fontsize=22)

    # 调整布局
    plt.tight_layout()

    if output_dir:
        output_path = output_dir / "unused_reagents_frequency_plot.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"\n图表已保存到: {output_path}")

    plt.show()

    return df


def main():
    # 定义路径（使用Pathlib）
    project_root = Path("/home/tzz/AIChem/synbo")

    # 描述符文件目录
    descriptor_dir = project_root / "benchmark" / "datasets" / "HTE_datasets" / "B-H_HTE" / "descriptors"

    # batch结果文件目录
    results_dir = project_root / "benchmark" / "compare_mothods" / "edboplus" / "results" / "EDBOplus_for_B-H_HTE"

    # 输出目录
    output_dir = project_root / "benchmark" / "compare_mothods" / "edboplus" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("试剂使用频率分析 (按batch统计未使用次数)")
    print("=" * 70)

    # 1. 获取所有可用试剂
    print("\n1. 从描述符文件中读取所有可用试剂...")
    all_reagents = get_all_reagents_from_descriptors(descriptor_dir)
    print(f"   Ligands: {len(all_reagents['ligand'])} 个")
    print(f"   Bases: {len(all_reagents['base'])} 个")
    print(f"   Solvents: {len(all_reagents['solvent'])} 个")

    # 2. 统计每个试剂在每个batch中未使用的次数
    print("\n2. 分析每个batch中的试剂使用情况...")
    unused_counts, total_batches = count_unused_per_batch(all_reagents, results_dir)

    # 3. 打印结果
    print("\n" + "=" * 70)
    print("统计结果")
    print("=" * 70)

    all_unused_list = []

    for reagent_type in ["ligand", "base", "solvent"]:
        print(f"\n【{reagent_type.upper()}】")
        print("-" * 50)

        # 按未使用次数排序
        sorted_reagents = sorted(unused_counts[reagent_type].items(), key=lambda x: x[1], reverse=True)

        for reagent, count in sorted_reagents:
            if count > 0:
                percentage = count / total_batches * 100
                print(f"  {reagent:25s} : 未使用 {count:2d}/{total_batches} 次")
                all_unused_list.append(
                    {
                        "Type": reagent_type.upper(),
                        "Reagent": reagent,
                        "Unused Count": count,
                        "Total Batches": total_batches,
                        "Unused Rate (%)": round(percentage, 1),
                    }
                )
            else:
                print(f"  {reagent:25s} : 已使用于所有 {total_batches} 个batch")

    # 4. 创建汇总列表
    print("\n" + "=" * 70)
    print("未使用试剂汇总列表（至少在一个batch中未使用）")
    print("=" * 70)

    if all_unused_list:
        print(f"\n共 {len(all_unused_list)} 个试剂在某些batch中未被使用:")
        for item in all_unused_list:
            print(
                f"  [{item['Type']}] {item['Reagent']:<25s} : "
                f"{item['Unused Count']}/{item['Total Batches']} ({item['Unused Rate (%)']}%)"
            )
    else:
        print("\n所有试剂在每个batch中都被使用了！")

    # 5. 绘制频数图
    print("\n" + "=" * 70)
    print("生成可视化图表...")
    print("=" * 70)
    df_result = plot_unused_reagents(unused_counts, total_batches, output_dir)

    print("\n" + "=" * 70)
    print("分析完成！")
    print("=" * 70)

    return unused_counts, df_result


if __name__ == "__main__":
    unused_counts, df_result = main()
