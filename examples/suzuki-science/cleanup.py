from quanda.utils import canonicalize_input_SMILES_file

# canonicalize_input_SMILES_file("suzuki-HTE.csv", "base")
import pandas as pd

df = pd.read_csv("suzuki-HTE.csv")
print(df["base"].drop_duplicates())
