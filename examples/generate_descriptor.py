from pathlib import Path
import pandas as pd
from rxnopt.descriptor.spoc_desc import calc_spoc_desc

dataset_name = "B-H_HTE"
mol_types = ["base", "ligand", "solvent"]

df = pd.read_csv(Path(dataset_name) / Path(f"{dataset_name}.csv"))
for mol_type in mol_types:
    smiles_list = df[mol_type].drop_duplicates()
    smiles_list = pd.read_csv(Path(dataset_name) / Path(f"descriptors/{mol_type}_dft.csv"))[f"{mol_type}_SMILES"]
    calc_spoc_desc(smiles_list=smiles_list, save_path=Path(dataset_name) / Path(f"descriptors/{mol_type}.csv"), fp_type="OneHot")
