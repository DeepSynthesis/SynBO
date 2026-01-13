#!/usr/bin/env python3
"""Test script for resave_output_results function."""

from pathlib import Path
from rxnopt.utils.export_data import resave_output_results

# Test with the sample CSV file
input_csv = Path("testfile/start_file.csv")
output_dir = Path("test_output")
output_dir.mkdir(exist_ok=True)

# Define condition columns and metrics columns
condition_columns = ["base", "ligand", "solvent", "concentration", "temperature"]
metrics_columns = ["yield", "cost"]

print("Testing CSV to CSV conversion...")
try:
    resave_output_results(
        str(input_csv), str(output_dir / "output.csv"), condition_columns=condition_columns, metrics_columns=metrics_columns
    )
    print("✓ CSV to CSV conversion successful")
except Exception as e:
    print(f"✗ CSV to CSV conversion failed: {e}")

print("\nTesting CSV to Excel conversion...")
try:
    resave_output_results(
        str(input_csv), str(output_dir / "output.xlsx"), condition_columns=condition_columns, metrics_columns=metrics_columns
    )
    print("✓ CSV to Excel conversion successful")
except Exception as e:
    print(f"✗ CSV to Excel conversion failed: {e}")

print("\nTesting CSV to JSON conversion...")
try:
    resave_output_results(
        str(input_csv), str(output_dir / "output.json"), condition_columns=condition_columns, metrics_columns=metrics_columns
    )
    print("✓ CSV to JSON conversion successful")
except Exception as e:
    print(f"✗ CSV to JSON conversion failed: {e}")

    resave_output_results(
        str(input_csv), str(output_dir / "output.json"), condition_columns=condition_columns, metrics_columns=metrics_columns
    )

print("\nAll tests completed!")
