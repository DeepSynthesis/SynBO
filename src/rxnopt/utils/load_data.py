from pathlib import Path
import pandas as pd


def load_desc_from_file(desc_file: str, idx_col: str = "SMILES") -> pd.DataFrame:
    desc_file = Path(desc_file)
    assert desc_file.exists(), f"Descriptor file {desc_file} does not exist!"
    try:
        if desc_file.suffix == ".csv":
            df = pd.read_csv(desc_file, index_col=idx_col)
        elif desc_file.suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(desc_file, index_col=idx_col)
        else:
            raise Exception(f"Unsupported descriptor file format: {desc_file.suffix}")
    except Exception as e:
        raise Exception(f"Error loading descriptor file {desc_file}: {e}. \nMaybe check if the index_col '{idx_col}' exists in the file.")

    return df


def _convert_tag(tag, length):
    if tag is None:
        tag = [None] * length
    elif type(tag) is str:
        tag = [tag] * length
    else:
        assert type(tag) is list and length, "index_col should be a string or a list with the same length as reagent_types."
    return tag


def load_desc_dict(
    reagent_types: list,
    desc_dir: list | str,
    name_suffix: list | str = None,
    index_col: str = "SMILES",
    return_condition_dict: bool = False,
) -> dict:
    index_col = _convert_tag(index_col, len(reagent_types))
    name_suffix = _convert_tag(name_suffix, len(reagent_types))
    desc_dict = {}
    desc_dir = Path(desc_dir)
    for r_type, idx_col, name_s in zip(reagent_types, index_col, name_suffix):
        desc_file = desc_dir / f"{r_type}_desc.csv"
        if name_s is not None:
            desc_file = desc_dir / f"{r_type}{name_s}.csv"
        else:
            desc_file = desc_dir / f"{r_type}_desc.csv"
        assert desc_file.exists(), f"Descriptor file `{desc_file}` for {r_type} does not exist in {desc_dir}."

        df = load_desc_from_file(desc_file, idx_col=idx_col)
        desc_dict[r_type] = df

    if return_condition_dict:
        condition_dict = {k: v.index.tolist() for k, v in desc_dict.items()}
        return desc_dict, condition_dict
    else:
        return desc_dict


def load_condition_dict(reagent_types: list, rxn_space_dir: str, index_col: str = None) -> dict:
    index_col = _convert_tag(index_col, len(reagent_types))
    condition_dict = {}
    rxn_space_dir = Path(rxn_space_dir)
    for idx_col, r_type in zip(reagent_types, index_col):
        rxn_space_file = rxn_space_dir / f"{r_type}.csv"
        assert rxn_space_file.exists(), f"Reaction space file for {r_type} does not exist in {rxn_space_dir}."
        df = pd.read_csv(rxn_space_file)
        if index_col is not None:
            assert idx_col in df.columns, f"Index column {idx_col} not found in {rxn_space_file}."
            df.set_index(idx_col, inplace=True)
        condition_dict[r_type] = df.index.tolist()
    return condition_dict


def get_prev_rxn(file_pattern: str = "results/batch-*.csv") -> pd.DataFrame:
    return pd.concat([pd.read_csv(f) for f in Path().parent.glob(file_pattern)])
