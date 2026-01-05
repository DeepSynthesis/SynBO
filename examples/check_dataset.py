import sys
import pandas as pd
from itertools import product

dataset_info = {
    "asym_hydrogenation": {
        "file_path": "asym_hydrogenation/asym_hydrogenation.csv",
        "reagent_columns": [
            "reagent",
            "solvent",
            "ligand",
            "metal_amount",
            "ligand_amount",
            "temperature",
            "pressure",
            "time",
        ],
    },
    "asym_alkylation": {
        "file_path": "asym_alkylation/asym_alkylation.csv",
        "reagent_columns": ["Reaction1", "Reaction2", "Catalyst1", "Catalyst2"],
    },
    "B-H_HTE": {"file_path": "B-H_HTE/B-H_HTE.csv", "reagent_columns": ["base", "ligand", "solvent", "concentration", "temperature"]},
    "suzuki_HTE": {
        "file_path": "suzuki_HTE/suzuki_HTE.csv",
        "reagent_columns": ["product", "solvent", "catalyst", "ligand", "reactant2", "reactant1", "base"],
    },
    "amide_coupling": {
        "file_path": "amide_coupling/amide_coupling.csv",
        "reagent_columns": ["solvent", "base", "activator", "nucleophile"],
    },
    "C-H_arylation": {
        "file_path": "C-H_arylation/C-H_arylation.csv",
        "reagent_columns": ["ligand", "electrophile", "nucleophile"],
    },
    "deoxyf": {
        "file_path": "deoxyf/deoxyf.csv",
        "reagent_columns": ["base", "fluoride", "substrate"],
    },
}


def process_reagent_data(file_path, reagent_cols):
    df = pd.read_csv(file_path)
    # df = df[df["reagent"] == "COC(=O)/C(=C/c1ccccc1)NC(C)=O"]
    reagent_list = []
    for col in reagent_cols:
        reagent_list.append(df[col].drop_duplicates())
    df = df.drop_duplicates(subset=reagent_cols)
    unique_values = [df[col].dropna().unique() for col in reagent_cols]
    total_combinations = 1
    for values in unique_values:
        total_combinations *= len(values)
    coverage_ratio = len(df) / total_combinations
    reagent_counts = {col: len(df[col].dropna().unique()) for col in reagent_cols}
    return df, coverage_ratio, reagent_counts


# Example usage
# dataset = "读取python的第一个参数"
dataset = sys.argv[1] if len(sys.argv) > 1 else "deoxyf"
file_path = dataset_info[dataset]["file_path"]
reagent_cols = dataset_info[dataset]["reagent_columns"]
result_df, coverage, counts = process_reagent_data(file_path, reagent_cols)
print(f"Coverage Ratio: {coverage}")
print("Reagent Counts:")
for reagent, count in counts.items():
    print(f"{reagent}: {count}")
