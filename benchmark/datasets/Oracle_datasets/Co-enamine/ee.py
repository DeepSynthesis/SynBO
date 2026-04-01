import pandas as pd

dict_df = pd.read_csv("co.csv")

d = {}
for k, v in zip(dict_df["Molecule"], dict_df["SMILES"]):
    d[k] = v

ori_df = pd.read_csv("descriptors/cobalt_desc_dft.csv")
ori_df["SMILES"] = ori_df["SMILES"].map(d)
ori_df.dropna(axis=0, inplace=True)
ori_df.reset_index(drop=True, inplace=True)
print(ori_df["SMILES"])
ori_df.to_csv("descriptors/cobalt_desc_dft.csv", index=False)
