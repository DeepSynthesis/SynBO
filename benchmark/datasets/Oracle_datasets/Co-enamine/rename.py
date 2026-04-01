import pandas as pd

dict_df = pd.read_csv("descriptors/mappings.csv")
d = {}
for info in dict_df["SMILES", "Abbreviation"]:
    d[info[0]] = info[1]

for desc in ["alkali", "amine", "cobalt", "oxidant", "solvent"]:
    df = pd.read_csv(f"descriptors/{desc}_desc.csv")
    df["SMILES"] = df["SMILES"].map(d)
    df.to_csv(f"descriptors/{desc}_desc.csv", index=False)

prev_rxn = pd.read_csv("Co-enamine.csv")
for col in ["alkali", "amine", "cobalt", "oxidant", "solvent"]:
    prev_rxn[f"{col}_smiles"] = prev_rxn[f"{col}_smiles"].map(d)

prev_rxn.to_csv("Co-enamine.csv", index=False)
