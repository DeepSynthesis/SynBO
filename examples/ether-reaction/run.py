import pandas as pd
from pathlib import Path
from rxnopt import ReactionOptimizer


if __name__ == "__main__":
    condition_types = ["additive", "amine_cat", "metal_cat", "oxidant", "solvent"]
    opt_metrics = ["yield", "ee"]
    # # load previous data
    # prev_rxn_list = [pd.read_csv(f) for f in Path(__file__).parent.glob("results/batch-*.csv")]
    # prev_rxn = pd.concat(prev_rxn_list, ignore_index=True)
    # load metrics
    condition_dict = {}
    for c in condition_types:
        df = pd.read_csv(Path(__file__).parent / Path(f"rxn_space/{c}.csv"))
        condition_dict[c] = df["SMILES"].drop_duplicates().tolist() if c != "metal_cat" else df["Molecule"].drop_duplicates().tolist()

    reaction_optimizer = ReactionOptimizer(opt_metrics=opt_metrics, opt_type="init")
    reaction_optimizer.load_rxn_space(condition_dict=condition_dict)
    reaction_optimizer.load_desc()
    # reaction_optimizer.load_prev_rxn(prev_rxn, drop_rxn=True)
    # reaction_optimizer.run(batch_size=3, desc_normalize="minmax")
    reaction_optimizer.initialize(batch_size=5, sampling_method="cvt")
    # reaction_optimizer.save_recommendations(Path(__file__).parent / Path("results/"))
