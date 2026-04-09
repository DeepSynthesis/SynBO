import pandas as pd
import os

# File paths
input_csv = (
    "/home/tzz/AIChem/synbo/benchmark/compare_mothods/edboplus/results/EDBOplus_for_asym_alkylation/merged_EDBOplus_for_asym_alkylation.csv"
)
hte_csv = "/home/tzz/AIChem/synbo/benchmark/datasets/HTE_datasets/asym_alkylation/asym_alkylation.csv"
output_dir = "/home/tzz/AIChem/synbo/benchmark/compare_mothods/edboplus/results"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Read CSV file
df_input = pd.read_csv(input_csv, usecols=["step", "sample_index", "round_id"])
df_hte = pd.read_csv(hte_csv)

# Rename step column to batch
df_input = df_input.rename(columns={"step": "batch"})

# All columns in HTE dataset（including yield and cost）
# hte_cols = ["base", "ligand", "solvent", "concentration", "temperature", "yield", "cost"]
hte_cols = ["reactant2", "catalyst1", "catalyst2", "yield", "ee"]

# Group by round_id，round_id range 1-10，corresponds to batch_0 to batch_9
for round_id in range(1, 11):
    # Filter current round_id data
    batch_data = df_input[df_input["round_id"] == round_id].copy()

    # Delete round_id column（Not needed in output）
    batch_data = batch_data.drop(columns=["round_id"])

    # Create HTE data columns
    for col in hte_cols:
        batch_data[col] = None

    # Match HTE data by sample_index
    for idx, row in batch_data.iterrows():
        sample_index = row["sample_index"]
        # Find corresponding rows in asym_alkylation.csv
        hte_row = df_hte[df_hte["index"] == sample_index]
        if not hte_row.empty:
            for col in hte_cols:
                batch_data.at[idx, col] = hte_row.iloc[0][col]

    # Generate output filename (round_id 1-10 corresponds to batch_0-9)
    output_file = os.path.join(output_dir, f"batch_{round_id - 1}.csv")

    # Save to file
    batch_data.to_csv(output_file, index=False)
    print(f"Saved {output_file} with {len(batch_data)} rows")

print("All batches have been split and saved successfully!")
