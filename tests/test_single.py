from synbo import ReactionOptimizer
from synbo.utils import load_desc_dict


reagent_types = ["base", "ligand", "solvent", "concentration", "temperature"]
name_suffix = ["_dft", "_dft", "_dft", None, None]
opt_direct_info = [{"opt_direct": "max", "opt_range": [0, 100]}, {"opt_direct": "min", "opt_range": [0, 0.5]}]  # cost(min), yield(max)

desc_dict, condition_dict = load_desc_dict(
    reagent_types=reagent_types,
    desc_dir="dataset/descriptors",
    name_suffix=name_suffix,
    return_condition_dict=True,
    index_col=reagent_types,
)


rxn_opt = ReactionOptimizer(opt_metrics=["yield", "cost"], opt_metric_settings=opt_direct_info, opt_type="auto")
rxn_opt.load_rxn_space(condition_dict=condition_dict)
rxn_opt.load_desc(desc_dict=desc_dict)


rxn_opt.initialize(batch_size=5, desc_normalize="minmax", sampling_method="kmeans", refine_desc="filter_0.8")
rxn_opt.save_results(save_dir="testfile", filetype="csv")  # test json output
