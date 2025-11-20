from pathlib import Path
from rxnopt.descriptor import calc_spoc_desc

test_filepath = Path(__file__).parent / "descriptor"


def test_spoc_desc():
    smiles_list = ["CCO", "CCN", "CS(=O)C"]
    save_path = test_filepath / "spoc_desc_results.csv"
    # calc_spoc_desc(smiles_list, save_path=save_path, fp_type="RDKit")
    for fp_type in [
        "OneHot",
        "RDKit",
        "RDKitFP",
        "RDKitLinear",
        "AtomPaires",
        "TopologicalTorsions",
        "MACCSKeys",
        "Morgan",
        "Avalon",
        "LayeredFingerprint",
        "Estate",
        "EstateIndices",
    ]:
        calc_spoc_desc(smiles_list, save_path=save_path, fp_type=fp_type, desc_type_to_filename=True)


if __name__ == "__main__":
    test_spoc_desc()
