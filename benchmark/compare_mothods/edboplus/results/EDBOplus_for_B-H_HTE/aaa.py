import pandas as pd
import os

# Read the merged CSV file
input_file = 'merged_EDBOplus_for_B-H_HTE.csv'
df = pd.read_csv(input_file)

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Rename columns - first rename round_id to batch_new to avoid conflict
df = df.rename(columns={
    'round_id': 'batch_new',
    'yield_collected_values': 'yield',
    'cost_collected_values': 'cost'
})

# Drop the original 'batch' column (column at index 3)
df = df.drop(df.columns[3], axis=1)

# Now rename batch_new to batch
df = df.rename(columns={'batch_new': 'batch'})

# Keep only the relevant columns we need
df = df[['step', 'batch', 'yield', 'cost']]

# Filter to keep only rows where step <= 9
df_filtered = df[df['step'] <= 9]

# Split by batch and save to separate CSV files
unique_batches = sorted(df_filtered['batch'].unique().tolist())

for batch_id in unique_batches:
    # Filter data for this batch
    batch_df = df_filtered[df_filtered['batch'] == batch_id]
    
    # Create output filename
    output_file = os.path.join(current_dir, f'batch_{batch_id}.csv')
    
    # Save to CSV
    batch_df.to_csv(output_file, index=False)
    print(f'Saved batch {batch_id} to {output_file}')

print(f'\nTotal batches processed: {len(unique_batches)}')