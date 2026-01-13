from pathlib import Path
import pandas as pd
from rxnopt.descriptor.spoc_desc import calc_spoc_desc

dataset_name = "asym_alkylation"
mol_types = ["reactant2", "catalyst1", "catalyst2"]

df = pd.read_csv(Path(dataset_name) / Path(f"{dataset_name}.csv"))
for mol_type in mol_types:
    calc_spoc_desc(
        smiles_list=df[mol_type].drop_duplicates(), save_path=Path(dataset_name) / Path(f"descriptors/{mol_type}.csv"), fp_type="RDKit"
    )
