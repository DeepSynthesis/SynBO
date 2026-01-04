from quanda.utils import canonicalize_input_SMILES_file


{
    "amide_coupling": {
        "file_path": "amide_coupling/amide_coupling.csv",
        "reagent_columns": ["solvent_name", "base_smiles", "solvent_smiles", "activator_smiles", "nucleophile_smiles"],
    },
    "C-H_arylation": {
        "file_path": "C-H_arylation/C-H_arylation.csv",
        "reagent_columns": ["ligand_smiles", "electrophile_smiles", "nucleophile_smiles"],
    },
    "deoxyf": {
        "file_path": "deoxyf/deoxyf.csv",
        "reagent_columns": ["base_smiles", "fluoride_smiles", "substrate_smiles"],
    },
}
canonicalize_input_SMILES_file(
    "deoxyf/deoxyf.csv",
    ["base_smiles", "fluoride_smiles", "substrate_smiles"],
)
