import pandas as pd
from pathlib import Path
from rxnopt import ReactionOptimizer



df = pd.read_csv(Path(__file__).parent / Path("1430-Ultra-high-throughput.csv"), encoding="UTF-8")
df["yield"] = df["yield"] / df["yield"].max()
condition_types = ["Reaction1", "Reaction2", "Catalyst1", "Catalyst2"]
opt_metrics = ["yield", "ee"]  #
# load previous data
prev_rxn_list = [pd.read_csv(f) for f in Path(__file__).parent.glob("results/batch-*.csv")]
prev_rxn = pd.concat(prev_rxn_list, ignore_index=True)
# load metrics
# for rxn in prev_rxn:
for idx, rxn_row in prev_rxn.iterrows():
    mask = (df[condition_types] == rxn_row[condition_types]).all(axis=1)
    matched_rows = df[mask]
    if not matched_rows.empty:
        matched_row = matched_rows.iloc[0]
        prev_rxn.loc[idx, opt_metrics] = matched_row[opt_metrics].values
prev_rxn["ee"] = prev_rxn["ee"].abs()

condition_dict = {c: df[c].drop_duplicates().tolist() for c in condition_types}
reaction_optimizer = ReactionOptimizer(opt_metrics=opt_metrics, opt_type="opt")
reaction_optimizer.load_rxn_space(condition_dict=condition_dict)
reaction_optimizer.load_desc()
reaction_optimizer.load_prev_rxn(prev_rxn, drop_rxn=True)
reaction_optimizer.run(batch_size=3, desc_normalize="minmax")
# reaction_optimizer.save_recommendations(Path(__file__).parent / Path("results/"))
