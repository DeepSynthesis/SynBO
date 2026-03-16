from pathlib import Path
import pandas as pd
from qspoc import QSPOCDesc

save_path = Path(__file__).parent / Path("..")

mol_type = "oxidant"
mol_df_path = Path(__file__).parent / Path(f"reagents/{mol_type}_smiles.csv")

qm_desc = QSPOCDesc(
    multiwfn_path="/home/tzz/AIChem/QSPOC/bin/Multiwfn/Multiwfn_3.8_bin_Linux_noGUI/Multiwfn_noGUI",
    save_dir=save_path,
    exe_path_dict={"xtb": "xtb"},
)
qm_desc.load_data_from_file(mol_df_path, f"{mol_type}_smiles", atomlists_tag="atom_idx")
qm_desc.get_init_sturct()
qm_desc.geometric_opt(method="xtb")
qm_desc.singlepoint_energy(method="xtb")
qm_desc.descriptor_calc(save=True)
