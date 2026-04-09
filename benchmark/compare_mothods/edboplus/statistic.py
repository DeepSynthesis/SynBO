import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def get_all_reagents_from_descriptors(descriptor_dir: Path) -> dict:
    """
    Read all available reagent names from descriptor files

    Returns:
        dict: Keys are 'ligand', 'base', 'solvent', values are reagent name lists
    """
    reagents = {"ligand": [], "base": [], "solvent": []}

    # Read ligand descriptor (first column index is ligand name)
    ligand_file = descriptor_dir / "ligand_dft.csv"
    if ligand_file.exists():
        df = pd.read_csv(ligand_file, index_col=0)
        reagents["ligand"] = list(df.index)

    # Read base descriptor (first column index is base name)
    base_file = descriptor_dir / "base_dft.csv"
    if base_file.exists():
        df = pd.read_csv(base_file, index_col=0)
        reagents["base"] = list(df.index)

    # Read solvent descriptor (first column index is solvent name)
    solvent_file = descriptor_dir / "solvent_dft.csv"
    if solvent_file.exists():
        df = pd.read_csv(solvent_file, index_col=0)
        reagents["solvent"] = list(df.index)

    return reagents


def count_unused_per_batch(all_reagents: dict, results_dir: Path) -> dict:
    """
    Count unused times for each reagent

    Returns:
        dict: Key is reagent name, values are how many batches it was not used in
    """
    # Get all batch files
    batch_files = sorted(results_dir.glob("batch_*.csv"))
    total_batches = len(batch_files)

    # Initialize counter: times each reagent is unused
    unused_counts = {
        "ligand": {reagent: 0 for reagent in all_reagents["ligand"]},
        "base": {reagent: 0 for reagent in all_reagents["base"]},
        "solvent": {reagent: 0 for reagent in all_reagents["solvent"]},
    }

    print(f"\n   Found {total_batches} batch files")

    for batch_file in batch_files:
        df = pd.read_csv(batch_file)

        # Check ligand
        if "ligand" in df.columns:
            used_ligands = set(df["ligand"].dropna().unique())
            for ligand in all_reagents["ligand"]:
                if ligand not in used_ligands:
                    unused_counts["ligand"][ligand] += 1

        # Check base
        if "base" in df.columns:
            used_bases = set(df["base"].dropna().unique())
            for base in all_reagents["base"]:
                if base not in used_bases:
                    unused_counts["base"][base] += 1

        # Check solvent
        if "solvent" in df.columns:
            used_solvents = set(df["solvent"].dropna().unique())
            for solvent in all_reagents["solvent"]:
                if solvent not in used_solvents:
                    unused_counts["solvent"][solvent] += 1

    return unused_counts, total_batches


def plot_unused_reagents(unused_counts: dict, total_batches: int, output_dir: Path = None):
    """
    Use seaborn to plot frequency of unused reagents
    All reagents on one figure, same color for same type
    """
    # Set Arial font
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    # Assign colors for each reagent type
    type_colors = {"Ligand": "#1f77b4", "Base": "#ff7f0e", "Solvent": "#2ca02c"}  # Blue  # Orange  # Green

    # Prepare data: filter out reagents never unused (count is 0)
    data = []
    for reagent_type, counts in unused_counts.items():
        for reagent, count in counts.items():
            if count > 0:  # only show reagents with unused records
                data.append(
                    {
                        "Reagent Type": reagent_type.capitalize(),
                        "Reagent": reagent,
                        "Unused Count": count,
                        "Unused Rate": count / total_batches * 100,
                    }
                )

    if not data:
        print("All reagents were used in every batch！")
        return None

    df = pd.DataFrame(data)

    # Sort by reagent type and unused count for display (Ligand first, then Base, Solvent)
    type_order = {"Ligand": 0, "Base": 1, "Solvent": 2}
    df["TypeOrder"] = df["Reagent Type"].map(type_order)
    df = df.sort_values(["TypeOrder", "Unused Count"], ascending=[True, True])

    # Create single figure
    fig, ax = plt.subplots(figsize=(16, 10))

    # Draw bar chart for each reagent, same type uses same color
    y_positions = range(len(df))
    bar_colors = [type_colors[rt] for rt in df["Reagent Type"]]
    bars = ax.barh(y_positions, df["Unused Count"], color=bar_colors, edgecolor="black", linewidth=0.5)

    # Set y-axis labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(df["Reagent"], fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)

    # Set title and labels
    ax.set_title(f"Unused Reagents Frequency (out of {total_batches} batches)", fontsize=20, fontweight="bold", pad=20)
    ax.set_xlabel(f"Number of Batches Not Used", fontsize=22)
    ax.set_ylabel("Reagent", fontsize=18)

    # Add value labels
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

    # Add grid lines
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    # Add legend for reagent types
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=type_colors["Ligand"], edgecolor="black", label=f'Ligand ({len(df[df["Reagent Type"]=="Ligand"])})'),
        Patch(facecolor=type_colors["Base"], edgecolor="black", label=f'Base ({len(df[df["Reagent Type"]=="Base"])})'),
        Patch(facecolor=type_colors["Solvent"], edgecolor="black", label=f'Solvent ({len(df[df["Reagent Type"]=="Solvent"])})'),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=20, title="Reagent Types", title_fontsize=22)

    # Adjust layout
    plt.tight_layout()

    if output_dir:
        output_path = output_dir / "unused_reagents_frequency_plot.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"\nChart saved to: {output_path}")

    plt.show()

    return df


def main():
    # Define paths (using Pathlib)
    project_root = Path("/home/tzz/AIChem/synbo")

    # Descriptor file directory
    descriptor_dir = project_root / "benchmark" / "datasets" / "HTE_datasets" / "B-H_HTE" / "descriptors"

    # Batch results file directory
    results_dir = project_root / "benchmark" / "compare_mothods" / "edboplus" / "results" / "EDBOplus_for_B-H_HTE"

    # Output directory
    output_dir = project_root / "benchmark" / "compare_mothods" / "edboplus" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Reagent usage frequency analysis (count unused per batch)")
    print("=" * 70)

    # 1. Get all available reagents
    print("\n1. from descriptor file read all available reagents...")
    all_reagents = get_all_reagents_from_descriptors(descriptor_dir)
    print(f"   Ligands: {len(all_reagents['ligand'])} ")
    print(f"   Bases: {len(all_reagents['base'])} ")
    print(f"   Solvents: {len(all_reagents['solvent'])} ")

    # 2. Count unused times for each reagent
    print("\n2. analyze in each batch reagent usage...")
    unused_counts, total_batches = count_unused_per_batch(all_reagents, results_dir)

    # 3. Print results
    print("\n" + "=" * 70)
    print("Statistics")
    print("=" * 70)

    all_unused_list = []

    for reagent_type in ["ligand", "base", "solvent"]:
        print(f"\n【{reagent_type.upper()}】")
        print("-" * 50)

        # sort by unused count
        sorted_reagents = sorted(unused_counts[reagent_type].items(), key=lambda x: x[1], reverse=True)

        for reagent, count in sorted_reagents:
            if count > 0:
                percentage = count / total_batches * 100
                print(f"  {reagent:25s} : unused {count:2d}/{total_batches} times")
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
                print(f"  {reagent:25s} : used in all {total_batches} batches")

    # 4. Create summary list
    print("\n" + "=" * 70)
    print("unused reagent summary list (at least unused in some batches)")
    print("=" * 70)

    if all_unused_list:
        print(f"
Total {len(all_unused_list)} reagents unused in some batches:")
        for item in all_unused_list:
            print(
                f"  [{item['Type']}] {item['Reagent']:<25s} : "
                f"{item['Unused Count']}/{item['Total Batches']} ({item['Unused Rate (%)']}%)"
            )
    else:
        print("\nAll reagents were used in every batch！")

    # 5. Draw frequency plot
    print("\n" + "=" * 70)
    print("Generating visualization charts...")
    print("=" * 70)
    df_result = plot_unused_reagents(unused_counts, total_batches, output_dir)

    print("\n" + "=" * 70)
    print("Analysis complete！")
    print("=" * 70)

    return unused_counts, df_result


if __name__ == "__main__":
    unused_counts, df_result = main()
