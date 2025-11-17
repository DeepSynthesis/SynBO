from pathlib import Path
from rxnopt.descriptor import calc_spoc_desc

test_filepath = Path(__file__).parent / "data"


def test_spoc_desc():
    smiles_list = ["CCO", "CCN", "CS(=O)C"]
    save_path = test_filepath / "spoc_desc_results.csv"
