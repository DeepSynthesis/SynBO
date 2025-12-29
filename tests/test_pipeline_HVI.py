from datetime import datetime
import os
from pathlib import Path
import shutil
from matplotlib import pyplot as plt
import pandas as pd
from rxnopt import ReactionOptimizer
from rxnopt.descriptor.spoc_desc import calc_spoc_desc
from rxnopt.utils import load_desc_dict, get_prev_rxn

import seaborn as sns
import numpy as np


def fill_done_dir(i, date):
    current_df = pd.read_csv(f"results/batch-{i}_{date}.csv")
    current_df.drop(columns=["hvi"], inplace=True)
    HTE_df = pd.read_csv(f"dataset/B-H_dataset.csv")
    merged_df = pd.merge(
        current_df,
        HTE_df[["base", "ligand", "solvent", "concentration", "temperature", "yield", "cost", "hvi"]],
        on=["base", "ligand", "solvent", "concentration", "temperature"],
        how="left",
    )
    merged_df.to_csv(f"results/batch-{i}_{date}.csv", index=False)


date = datetime.now().strftime("%Y%m%d")
for f in Path("results/").glob(f"batch-*.csv"):
    os.remove(f)


# def generate_onehot():
#     for name in ["base", "ligand", "solvent"]:
#         df = pd.read_csv(f"dataset/descriptors/{name}_dft.csv")
#         calc_spoc_desc(df[f"{name}_file_name"], save_path=f"dataset/descriptors/{name}.csv", fp_type="RDKit", desc_type_to_filename=True)


# generate_onehot()
# df = pd.read_csv("dataset/B-H_dataset.csv")
# df["hvi"] = df.apply(lambda row: row["yield"] * (-row["cost"] + 0.5), axis=1)
# df.to_csv("dataset/B-H_dataset.csv", index=False)

# exit()
reagent_types = ["base", "ligand", "solvent", "concentration", "temperature"]
index_col = [f"{r}_file_name" for r in reagent_types]
name_suffix = ["_dft", "_dft", "_dft", None, None]
opt_direct_info = [{"opt_direct": "max", "opt_range": [0, 44.69]}]  # cost(min), yield(max)

desc_dict, condition_dict = load_desc_dict(
    reagent_types=reagent_types, desc_dir="dataset/descriptors", name_suffix=name_suffix, return_condition_dict=True, index_col=index_col
)

for i in range(20):
    rxn_opt = ReactionOptimizer(opt_metrics=["hvi"], opt_metric_settings=opt_direct_info, opt_type="auto")
    rxn_opt.load_rxn_space(condition_dict=condition_dict)
    rxn_opt.load_desc(desc_dict=desc_dict)
    if i > 0:
        rxn_opt.load_prev_rxn(prev_rxn_info=get_prev_rxn(file_pattern=f"results/batch-*.csv"))
    if i == 0:
        rxn_opt.initialize(batch_size=5, desc_normalize="minmax", sampling_method="cvt", refine_desc="filter_0.8")
    else:
        rxn_opt.optimize(batch_size=5, desc_normalize="minmax", mc_num_samples=32, max_batch_size=32, refine_desc="filter_0.8")
    rxn_opt.save_results(save_dir="results")

    fill_done_dir(i, date)

dfs = [pd.read_csv(f) for f in Path("results/").glob(f"batch-*.csv")]
df = pd.concat(dfs, ignore_index=True)
df.to_csv(f"results/all_batches_{date}.csv", index=False)
