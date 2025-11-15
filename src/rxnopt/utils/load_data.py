from pathlib import Path
import pandas as pd
from rich.console import Console

console = Console()


def load_desc_from_file(desc_file: str, index_col: str = None) -> pd.DataFrame:
    desc_file = Path(desc_file)
    if not desc_file.exists():
        console.print(f"Descriptor file {desc_file} does not exist!", style="bold red")
        raise Exception(f"Descriptor file {desc_file} does not exist!")

    if desc_file.suffix() == ".csv":
        df = pd.read_csv(desc_file, index_col=index_col)
    elif desc_file.suffix() in [".xlsx", ".xls"]:
        df = pd.read_excel(desc_file, index_col=index_col)
    else:
        console.print(f"Unsupported descriptor file format: {desc_file.suffix}", style="bold red")
        raise Exception(f"Unsupported descriptor file format: {desc_file.suffix}")

    return df


def load_desc_dict(reagent_types: list, desc_dir: str, index_col: str = None, return_rxn_space: bool = False) -> dict:
    if index_col is None:
        index_col = [None] * len(reagent_types)
    elif type(index_col) is str:
        index_col = [index_col] * len(reagent_types)
    else:
        assert type(index_col) is list and len(index_col) == len(
            reagent_types
        ), "index_col should be a string or a list with the same length as reagent_types."

    desc_dict = {}
    desc_dir = Path(desc_dir)
    for r_type in reagent_types:
        desc_file = desc_dir / f"{r_type}_desc.csv"
        assert desc_file.exists(), f"Descriptor file for {r_type} does not exist in {desc_dir}."
        df = load_desc_from_file(desc_file, index_col=index_col)
        desc_dict[r_type] = df
    
    if return_rxn_space:
        rxn_space_dict = {}
        
    
    return desc_dict
