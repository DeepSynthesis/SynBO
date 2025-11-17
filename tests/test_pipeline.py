import pandas as pd
from rxnopt import ReactionOptimizer
from rxnopt.utils import load_desc_dict, get_prev_rxn


def fill_done_dir(i):
    current_df = pd.read_csv(f"results/batch-{i}_20251117.csv")
    HTE_df = pd.read_csv(f"dataset/B-H_dataset.csv")
    merged_df = pd.merge(current_df, HTE_df[["base", "ligand", "solvent", "yield", "cost"]], on=["base", "ligand", "solvent"], how="left")
    merged_df.to_csv(f"results/batch-{i}_20251117.csv", index=False)


reagent_types = ["base", "ligand", "solvent", "concentration", "temperature"]
index_col = [f"{r}_file_name" for r in reagent_types]
name_suffix = ["_dft", "_dft", "_dft", None, None]

rxn_opt = ReactionOptimizer(opt_metrics=["yield", "cost"])
desc_dict, condition_dict = load_desc_dict(
    reagent_types=reagent_types, desc_dir="dataset/descriptors", name_suffix=name_suffix, return_condition_dict=True, index_col=index_col
)
condition_dict["concentration"] = pd.DataFrame({"concentration": [0.1, 0.2, 0.3]}).index.tolist()

for i in range(30):
    if i > 0:
        rxn_opt.load_prev_rxn(prev_rxn_info=get_prev_rxn(file_pattern=f"results/batch-*.csv"))
    rxn_opt.load_rxn_space(condition_dict=condition_dict)
    rxn_opt.load_desc(desc_dict=desc_dict)
    rxn_opt.run()
    rxn_opt.save_results(save_dir="results")

    fill_done_dir(i)
