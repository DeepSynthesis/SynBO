#!/usr/bin/env python3
"""Test script for resave_output_results function."""

from pathlib import Path
from rxnopt.utils.util_func import resave_output_results

# Test with the sample CSV file
input_csv = Path("tests/testfile/start_file.csv")
output_dir = Path("test_output")
output_dir.mkdir(exist_ok=True)

print("Testing CSV to CSV conversion...")
try:
    resave_output_results(str(input_csv), str(output_dir / "output.csv"))
    print("✓ CSV to CSV conversion successful")
except Exception as e:
    print(f"✗ CSV to CSV conversion failed: {e}")

print("\nTesting CSV to Excel conversion...")
try:
    resave_output_results(str(input_csv), str(output_dir / "output.xlsx"))
    print("✓ CSV to Excel conversion successful")
except Exception as e:
    print(f"✗ CSV to Excel conversion failed: {e}")

print("\nTesting CSV to JSON conversion...")
try:
    resave_output_results(str(input_csv), str(output_dir / "output.json"))
    print("✓ CSV to JSON conversion successful")
except Exception as e:
    print(f"✗ CSV to JSON conversion failed: {e}")

print("\nAll tests completed!")
