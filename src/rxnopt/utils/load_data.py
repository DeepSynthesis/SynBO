from pathlib import Path
import pandas as pd


def load_desc_from_file(desc_file: str, idx_col: str = "SMILES", fillna: bool = False) -> pd.DataFrame:
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

    if fillna:
        df.fillna(0.0, inplace=True)
    else:
        assert not df.isna().any().any(), f"Descriptor file `{desc_file}` contains NaN values. Please check the data."

    if df.index.duplicated().any():
        duplicated_items = df.index[df.index.duplicated()].unique().tolist()
        raise ValueError(f"Descriptor file {desc_file} contains duplicated molecules: {duplicated_items}")

    df.index = df.index.astype("str")
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
    name_suffix: list | str = None,  # TODO: do not set as None, must be _desc
    index_col: str = "SMILES",
    return_condition_dict: bool = False,
    fillna: bool = False,
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

        desc_dict[r_type] = load_desc_from_file(desc_file, idx_col=idx_col, fillna=fillna)

    if return_condition_dict:
        condition_dict = {str(k): v.index.astype("str").tolist() for k, v in desc_dict.items()}
        return desc_dict, condition_dict
    else:
        return desc_dict


def load_condition_dict(reagent_types: list, rxn_space_dir: str, index_col: str = None, value_col: str = None) -> dict:
    index_col = _convert_tag(index_col, len(reagent_types))
    value_col = _convert_tag(value_col, len(reagent_types))
    condition_dict = {}
    rxn_space_dir = Path(rxn_space_dir)
    for r_type, idx_col, v_col in zip(reagent_types, index_col, value_col):
        rxn_space_file = rxn_space_dir / f"{r_type}.csv"
        assert rxn_space_file.exists(), f"Reaction space file for `{r_type}` does not exist in `{rxn_space_dir}`."
        df = pd.read_csv(rxn_space_file)
        if "select" in df.columns:
            df = df[df["select"]]

        if idx_col is not None:
            assert idx_col in df.columns, f"Index column {idx_col} not found in {rxn_space_file}."
            df.set_index(idx_col, inplace=True)

        if v_col is not None:
            condition_dict[r_type] = [{k: v} for k, v in zip(df.index.tolist(), df[v_col].tolist())]
        else:
            condition_dict[r_type] = df.index.astype("str").tolist()

    return condition_dict


def get_prev_rxn(file_root_dir: str = ".", file_pattern: str = "results/batch-*.csv") -> pd.DataFrame:
    iter_file = Path(file_root_dir).glob(file_pattern)
    assert len(list(iter_file)) > 0, f"There are no file with `{file_pattern}`."
    return pd.concat([pd.read_csv(f) for f in Path(file_root_dir).glob(file_pattern)])
