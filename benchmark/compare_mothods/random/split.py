import pandas as pd
import os

# File paths
input_csv = "/home/tzz/AIChem/synbo/benchmark/compare_mothods/random/results/all_results_Random_for_suzuki_HTE.csv"
hte_csv = "/home/tzz/AIChem/synbo/benchmark/datasets/HTE_datasets/suzuki_HTE/suzuki_HTE.csv"
output_dir = "/home/tzz/AIChem/synbo/benchmark/compare_mothods/random/results/suzuki_HTE"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Read CSV files
df_input = pd.read_csv(input_csv, usecols=["sample_index", "round_id", "step"])
df_hte = pd.read_csv(hte_csv)

# Rename step column to batch
df_input = df_input.rename(columns={"step": "batch"})

# All columns from the HTE dataset (including yield and cost)
hte_cols = ["solvent", "ligand", "reactant2", "reactant1", "base", "Conversion"]
# hte_cols = ["base", "ligand", "solvent", "concentration", "temperature", "yield", "cost"]
# hte_cols = ["reactant2", "catalyst1", "catalyst2", "yield", "ee"]

# Group by round_id (1-10), mapping to batch_0 through batch_9
for round_id in range(1, 11):
    # Filter data for current round_id
    batch_data = df_input[df_input["round_id"] == round_id].copy()

    # Remove round_id column (not needed in output)
    batch_data = batch_data.drop(columns=["round_id"])

    # Create HTE data columns
    for col in hte_cols:
        batch_data[col] = None

    # Match HTE data by sample_index
    for idx, row in batch_data.iterrows():
        sample_index = row["sample_index"]
        # Look up corresponding row in HTE csv
        hte_row = df_hte[df_hte["index"] == sample_index]
        if not hte_row.empty:
            for col in hte_cols:
                batch_data.at[idx, col] = hte_row.iloc[0][col]

    # Generate output filename (round_id 1-10 maps to batch_0-9)
    output_file = os.path.join(output_dir, f"batch_{round_id - 1}.csv")

    # Save to file
    batch_data.to_csv(output_file, index=False)
    print(f"Saved {output_file} with {len(batch_data)} rows")

print("All batches have been split and saved successfully!")
