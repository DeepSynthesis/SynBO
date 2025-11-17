from pathlib import Path
from typing import List, Optional
from quanda import MultiMolDesc
from rich.console import Console

console = Console()


def calc_qm_desc(
    smiles_list: List[str],
    save_path: Path | str,
    tag_list: List[str] = None,
    atom_lists: List[List[int]] = None,
    bond_lists: List[List[int]] = None,
    sterimol_vec_lists: List[List[int]] = None,
    ignore_IF: Optional[bool] = False,
) -> None:
    mmdesc = MultiMolDesc(ignore_IF=ignore_IF)
    mmdesc.load_data(smiles_list, tag_list, atom_lists, bond_lists, sterimol_vec_lists)
    mmdesc.compute_all_descriptors()
    mmdesc.save_results(save_path)
    failed_mols = mmdesc.get_failed_molecules()
    if failed_mols:
        console.print(f"Error: some molecules failed to be calculated:" f"{failed_mols}", style="red")


def calc_qm_desc_from_file(
    file_path: str,
    save_path: str,
    smiles_tag: str = "SMILES",
    tag_tag: Optional[str] = None,
    atomlists_tag: Optional[str] = None,
    bondlists_tag: Optional[str] = None,
    stlists_tag: Optional[str] = None,
    prop_tag: Optional[str] = None,
    ignore_IF: Optional[bool] = False,
):
    mmdesc = MultiMolDesc(ignore_IF=ignore_IF)
    mmdesc.load_data_from_file(file_path, smiles_tag, tag_tag, atomlists_tag, bondlists_tag, stlists_tag, prop_tag)
    mmdesc.compute_all_descriptors()
    mmdesc.save_results(save_path)
    failed_mols = mmdesc.get_failed_molecules()
    if failed_mols:
        console.print(f"Error: some molecules failed to be calculated:" f"{failed_mols}", style="red")
