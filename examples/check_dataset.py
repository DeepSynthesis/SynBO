import pandas as pd
from itertools import product


def process_reagent_data(file_path, reagent_cols):
    df = pd.read_csv(file_path)
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
file_path = "1430-ultra-HTE/1430-Ultra-high-throughput.csv"
reagent_cols = ["Reaction1", "Reaction2", "Catalyst1", "Catalyst2"]
result_df, coverage, counts = process_reagent_data(file_path, reagent_cols)
print(f"Coverage Ratio: {coverage}")
print("Reagent Counts:")
for reagent, count in counts.items():
    print(f"{reagent}: {count}")
