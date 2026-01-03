import pandas as pd
from itertools import product

dataset_info = {
    "asym-hydrogenation": {
        "file_path": "asym-hydrogenation/asym_hydrogenation.csv",
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
    "1430-UHTE": {
        "file_path": "1430-ultra-HTE/1430-Ultra-high-throughput.csv",
        "reagent_columns": ["Reaction1", "Reaction2", "Catalyst1", "Catalyst2"],
    },
    "B-H-HTE": {"file_path": "B-H-HTE/B-H-dataset.csv", "reagent_columns": ["base", "ligand", "solvent", "concentration", "temperature"]},
    "suzuki": {
        "file_path": "suzuki-science/suzuki-HTE.csv",
        "reagent_columns": ["product", "solvent", "catalyst", "ligand", "reactant2", "reactant1", "base"],
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
    print(len(df), total_combinations)
    coverage_ratio = len(df) / total_combinations
    reagent_counts = {col: len(df[col].dropna().unique()) for col in reagent_cols}
    return df, coverage_ratio, reagent_counts


# Example usage
dataset = "suzuki"
file_path = dataset_info[dataset]["file_path"]
reagent_cols = dataset_info[dataset]["reagent_columns"]
result_df, coverage, counts = process_reagent_data(file_path, reagent_cols)
print(f"Coverage Ratio: {coverage}")
print("Reagent Counts:")
for reagent, count in counts.items():
    print(f"{reagent}: {count}")
