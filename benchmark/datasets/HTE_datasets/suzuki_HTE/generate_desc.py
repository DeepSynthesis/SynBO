from synbo.descriptor import calc_spoc_desc
import pandas as pd

df = pd.read_csv("suzuki_HTE.csv")
for col in ["solvent", "ligand", "reactant2", "reactant1", "base"]:
    l = df[col].drop_duplicates().tolist()
    if "blank_cell" in l:
        print(col)
    l.remove("blank_cell") if "blank_cell" in l else None

    calc_spoc_desc(l, f"descriptor/{col}.csv", l)
