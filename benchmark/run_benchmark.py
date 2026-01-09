import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from rxnopt import ReactionOptimizer
from rxnopt.utils import load_desc_dict, get_prev_rxn
from plots import plot_optimization_process, plot_hv_percentage

# =================================================CONFIG=================================================
global_dir = Path(__file__).parent
data_dir = global_dir / Path("../examples/")
CONFIG = {
    "experiment_name": "B-H_Optimization",
    "seed": 199,
    "iterations": 10,
    "batch_size": 5,
    "data_paths": {
        "dataset_file": str(data_dir / "B-H_HTE/B-H_HTE.csv"),
        "descriptor_dir": str(data_dir / "B-H_HTE/descriptors"),
        "results_base_dir": str(global_dir / "results"),
    },
    "reaction_space": {
        "reagent_types": ["base", "ligand", "solvent", "concentration", "temperature"],
        "name_suffix": ["_dft", "_dft", "_dft", None, None],
    },
    "optimization_settings": {
        "opt_metrics": ["cost", "yield"],  # ,
        "opt_direct_info": [
            {"opt_direct": "max", "opt_range": [0, 100], "metric_weight": 1.0},  # yield
            {"opt_direct": "min", "opt_range": [0, 0.5], "metric_weight": 1.0},  # cost
        ],
        "opt_type": "auto",
        "desc_normalize": "minmax",
        "sampling_method": "lhs",
        "refine_desc": "filter_0.8",
        "optimize_method": "evolution",
        "kwargs": {
            # "acq_func": "NEI",
            # "surrogate_model": "GP",
            "method": "Standard",
            "method": "RF",
        },
    },
}

CONFIG["reaction_space"]["index_col"] = [f"{r}_file_name" for r in CONFIG["reaction_space"]["reagent_types"]]
# ===========================================================================================================


def fill_done_dir(batch_idx, output_dir, dataset_path):
    """get yield and cost from dataset and fill into the saved batch file."""
    file_path = output_dir / f"batch-{batch_idx}_{datetime.now().strftime('%Y%m%d')}.csv"
    current_df = pd.read_csv(file_path)

    cols_to_drop = [c for c in ["yield", "cost"] if c in current_df.columns]
    current_df.drop(columns=cols_to_drop, inplace=True)

    hte_df = pd.read_csv(dataset_path)

    match_cols = ["base", "ligand", "solvent", "concentration", "temperature"]
    merged_df = pd.merge(
        current_df,
        hte_df[match_cols + ["yield", "cost"]],
        on=match_cols,
        how="left",
    )
    merged_df.to_csv(file_path, index=False)


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(CONFIG["data_paths"]["results_base_dir"]) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=4, ensure_ascii=False)

    desc_dict, condition_dict = load_desc_dict(
        reagent_types=CONFIG["reaction_space"]["reagent_types"],
        desc_dir=CONFIG["data_paths"]["descriptor_dir"],
        name_suffix=CONFIG["reaction_space"]["name_suffix"],
        index_col=CONFIG["reaction_space"]["index_col"],
        return_condition_dict=True,
    )

    for i in tqdm(range(CONFIG["iterations"]), desc="Overall Optimization Progress"):

        rxn_opt = ReactionOptimizer(
            opt_metrics=CONFIG["optimization_settings"]["opt_metrics"],
            opt_metric_settings=CONFIG["optimization_settings"]["opt_direct_info"],
            opt_type=CONFIG["optimization_settings"]["opt_type"],
            random_seed=CONFIG["seed"],
            quiet=False,
        )

        rxn_opt.load_rxn_space(condition_dict=condition_dict)
        rxn_opt.load_desc(desc_dict=desc_dict)

        if i > 0:
            prev_rxns = get_prev_rxn(file_root_dir=run_dir, file_pattern=str("batch-*.csv"))
            rxn_opt.load_prev_rxn(prev_rxn_info=prev_rxns)

        if i == 0:
            rxn_opt.initialize(
                batch_size=CONFIG["batch_size"],
                desc_normalize=CONFIG["optimization_settings"]["desc_normalize"],
                sampling_method=CONFIG["optimization_settings"]["sampling_method"],
                refine_desc=CONFIG["optimization_settings"]["refine_desc"],
            )
        else:
            rxn_opt.optimize(
                batch_size=CONFIG["batch_size"],
                optimize_method=CONFIG["optimization_settings"]["optimize_method"],
                desc_normalize=CONFIG["optimization_settings"]["desc_normalize"],
                refine_desc=CONFIG["optimization_settings"]["refine_desc"],
                **CONFIG["optimization_settings"]["kwargs"],
            )

        rxn_opt.save_results(save_dir=str(run_dir))

        fill_done_dir(i, run_dir, CONFIG["data_paths"]["dataset_file"])

    all_batch_files = sorted(list(run_dir.glob("batch-*.csv")))
    dfs = [pd.read_csv(f) for f in all_batch_files]
    df_all = pd.concat(dfs, ignore_index=True)
    df_all.to_csv(run_dir / "all_batches_final.csv", index=False)

    plot_optimization_process(root_dir=run_dir, file_pattern=str("batch-*.csv"), save_path=run_dir / "optimization_process.png")
    plot_hv_percentage(
        root_dir=run_dir,
        file_pattern=str("batch-*.csv"),
        dataset_path=CONFIG["data_paths"]["dataset_file"],
        opt_direct_info=CONFIG["optimization_settings"]["opt_direct_info"],
        save_path=run_dir / "hv_percentage.png",
    )


if __name__ == "__main__":
    main()
