import pandas as pd
from pathlib import Path
from rxnopt import ReactionOptimizer


def load_reaction_space():
    pass


if __name__ == "__main__":
    df = pd.read_csv(Path(__file__).parent / Path("../dataset/1430-Ultra-high-throughput.csv"))
    condition_types = ["Reaction1", "Reaction2", "Catalyst1", "Catalyst2"]
    condition_dict = {c: df[c].drop_duplicates().tolist() for c in condition_types}
    
    print(condition_dict)
    
    
    reaction_optimizer = ReactionOptimizer(opt_metrics=["yield", "ee"])
    reaction_optimizer.load_rxn_space(condition_dict=condition_dict)
    reaction_optimizer.load_desc()
    reaction_optimizer.run(batch_size=5, desc_normalize="zscore")
    
