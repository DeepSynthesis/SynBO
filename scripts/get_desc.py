#!/usr/bin/env python3
"""
Generate molecular descriptors from SMILES strings.

This script calculates SPOC (Structure-Property Oriented Classification) descriptors
for a list of SMILES strings and saves them to a CSV file.

Supports handling of entries without valid SMILES (e.g., "no_alkali") by filling
their descriptor columns with 0.0 placeholders.
"""

import argparse
from pathlib import Path
import pandas as pd
from typing import List, Tuple

try:
    from synbo.descriptor import spoc_desc
    from rdkit import Chem
except ImportError:
    print("Error: synbo/rdkit package is not installed. Please install it first.")
    raise Exception()


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate SPOC molecular descriptors from SMILES strings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate descriptors from a file containing SMILES (one per line)
  python get_desc.py --input reagent.csv --name "reagent" --output ./desc
        """,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", type=Path, help="Path to CSV file with SMILES and name columns")
    parser.add_argument("--name", required=True, help="Name for the output file (without extension)")
    parser.add_argument(
        "--output", type=Path, default=Path("./descriptors"),
        help="Output directory for the descriptor CSV file (default: ./descriptors)"
    )
    parser.add_argument("--index-name", default="Name", help="Name for the index column in the output CSV (default: Name)")
    parser.add_argument("--fp-type", default="RDKit", help="Fingerprint/descriptor type (default: RDKit)")

    return parser.parse_args()


def is_valid_smiles(smiles: str) -> bool:
    """Check if a SMILES string can be parsed by RDKit."""
    if not isinstance(smiles, str) or not smiles.strip():
        return False
    mol = Chem.MolFromSmiles(smiles.strip())
    return mol is not None


def read_smiles_from_file(file_path: Path) -> Tuple[pd.DataFrame, List[int]]:
    """Read SMILES and name from CSV. Returns (df, list of invalid SMILES row indices)."""
    assert file_path.exists(), f"Error: File {file_path} does not exist."
    df = pd.read_csv(file_path)
    assert "SMILES" in df.columns, f"Error: File {file_path} does not contain a 'SMILES' column."

    invalid_indices = []
    for idx, row in df.iterrows():
        if not is_valid_smiles(row["SMILES"]):
            invalid_indices.append(idx)

    return df, invalid_indices


def generate_descriptors(
    smiles_list: List[str], output_dir: Path, base_name: str,
    fp_type: str, index_name: str = "Name"
) -> pd.DataFrame:
    """
    Generate SPOC descriptors and return the result DataFrame.
    The actual saved file will be named {base_name}_{fp_type}.csv
    """
    save_path = output_dir / f"{base_name}.csv"
    try:
        print(f"Generating {fp_type} descriptors for {len(smiles_list)} SMILES strings...")
        spoc_desc.calc_spoc_desc(
            smiles_list=smiles_list,
            save_path=save_path,
            fp_type=fp_type,
            index_name=index_name,
        )
        # The actual file saved is {base_name}_{fp_type}.csv
        actual_path = output_dir / f"{base_name}_{fp_type}.csv"
        print(f"✅ Saved to: {actual_path}")
        result_df = pd.read_csv(actual_path, index_col=0)
        return result_df
    except Exception as e:
        raise RuntimeError(f"Failed to generate descriptors: {e}") from e


def main() -> int:
    args = parse_arguments()
    df, invalid_indices = read_smiles_from_file(args.input)

    if invalid_indices:
        invalid_names = df.loc[invalid_indices, "name"].tolist()
        print(f"⚠️  Found {len(invalid_indices)} entries with invalid SMILES (will use 0.0 placeholders):")
        for name in invalid_names:
            print(f"    - {name}")

    # Separate valid and invalid entries
    valid_df = df.drop(index=invalid_indices).reset_index(drop=True)
    
    if valid_df.empty:
        print("⚠️  No valid SMILES found. Nothing to generate.")
        return 0

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate descriptors for valid SMILES
    result_df = generate_descriptors(
        valid_df["SMILES"].tolist(), output_dir, args.name, args.fp_type, args.index_name
    )

    # Handle invalid SMILES: fill with 0.0 placeholders
    if invalid_indices:
        descriptor_cols = [c for c in result_df.columns if c != args.index_name]
        placeholder_rows = []
        for idx in invalid_indices:
            row = {args.index_name: df.loc[idx, "name"]}
            for col in descriptor_cols:
                row[col] = 0.0
            placeholder_rows.append(row)
        
        placeholder_df = pd.DataFrame(placeholder_rows)
        result_df = pd.concat([result_df, placeholder_df], ignore_index=True)
        
        actual_path = output_dir / f"{args.name}_{args.fp_type}.csv"
        result_df.to_csv(actual_path, index=False)
        print(f"✅ Added {len(invalid_indices)} placeholder entries (0.0-filled) to: {actual_path}")

    return 0


if __name__ == "__main__":
    main()
