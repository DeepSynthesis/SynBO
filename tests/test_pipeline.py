from rxnopt import ReactionOptimizer
from rxnopt.utils import load_desc_dict

reagent_types = ["base", "ligand", "solvent"]
index_col = [f"{r}_file_name" for r in reagent_types]

rxn_opt = ReactionOptimizer(opt_metrics=["yield", "cost"])
desc_dict, condition_dict = load_desc_dict(
    reagent_types=reagent_types, desc_dir="dataset/descriptors", name_suffix="_dft", return_condition_dict=True, index_col=index_col
)
rxn_opt.load_rxn_space(condition_dict=condition_dict)
rxn_opt.load_desc(desc_dict=desc_dict)
rxn_opt.run()
rxn_opt.save_results(save_dir="results")
