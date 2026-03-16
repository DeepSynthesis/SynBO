"""Utility functions for reaction optimization.

Modern utility functions with rich progress bars and improved error handling.
"""

from __future__ import annotations

from functools import wraps
from typing import List


import numpy as np
import pandas as pd
import torch
from pathlib import Path
from indigo import Indigo
from indigo.renderer import IndigoRenderer
from rdkit import Chem
import re


def track_called(func):
    """Decorator to track if a method has been called.

    Args:
        func: Function to track

    Returns:
        Wrapped function that sets a tracking attribute
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        setattr(self, f"_{func.__name__}_called", True)
        return func(self, *args, **kwargs)

    return wrapper


def check_SMILES(smiles_string: str) -> bool:
    from rdkit import Chem

    smiles_list = smiles_string.split(",")
    valid_smiles_list, invalid_smiles_list = [], []
    for smi in smiles_list:
        try:
            assert Chem.MolFromSmiles(smi) is not None
        except Exception:
            invalid_smiles_list.append(smi)

    return len(invalid_smiles_list) == 0, invalid_smiles_list


def generate_onehot_desc(condition_dict):
    # TODO: use SPOC onehot descriptors
    desc_dict = {}
    for k, v in condition_dict.items():
        desc_dict[k] = pd.get_dummies(v).T
    return desc_dict


def check_desc_completeness(desc_dict, condition_dict):
    for k, v in desc_dict.items():
        for name in condition_dict[k]:
            if not name in v.index:
                raise ValueError(f"Missing values in {k} description: {name}")


def compute_hvi(new_point, pareto_front, ref_point):
    from botorch.utils.multi_objective.hypervolume import Hypervolume

    # 确保输入是 torch.Tensor 类型
    if not isinstance(new_point, torch.Tensor):
        new_point = torch.tensor(new_point, dtype=torch.float32)
    if not isinstance(pareto_front, torch.Tensor):
        pareto_front = torch.tensor(pareto_front, dtype=torch.float32)
    if not isinstance(ref_point, torch.Tensor):
        ref_point = torch.tensor(ref_point, dtype=torch.float32)

    # 计算超体积
    hv = Hypervolume(ref_point=ref_point)
    original_hv = hv.compute(pareto_front)

    # 添加新点后的超体积
    extended_front = torch.cat([pareto_front, new_point.unsqueeze(0)], dim=0)
    new_hv = hv.compute(extended_front)

    return new_hv - original_hv


def get_opt_type(opt: str) -> str:
    if opt == "opt":
        return "Optimization"
    elif opt == "init":
        return "Initialization"


def generate_constraint_mask(
    total_name_arr: np.ndarray,
    condition_types: List[str],
    constraints: dict,
) -> np.ndarray:
    """Generate constraint mask based on condition restrictions.

    Args:
        total_name_arr: Array of all condition combinations (shape: [n_combinations, n_conditions])
        condition_types: List of condition type names
        constraints: Dictionary of constraints {condition_type: [prohibited_values]}

    Returns:
        Boolean mask array where True indicates the combination satisfies constraints
    """
    if not constraints:
        return np.ones(len(total_name_arr), dtype=bool)

    mask = np.ones(len(total_name_arr), dtype=bool)

    for condition_type, prohibited_values in constraints.items():
        if condition_type not in condition_types:
            continue  # Skip invalid condition types

        # Get index of this condition type in the name array
        condition_idx = condition_types.index(condition_type)

        # Check which combinations have allowed values for this condition
        prohibited_values = set(prohibited_values)
        condition_values = total_name_arr[:, condition_idx]
        condition_mask = np.array([val not in prohibited_values for val in condition_values])

        # Update overall mask
        mask = mask & condition_mask

    assert sum(mask) > 0, "No valid combinations found with given constraints"

    return mask


def sanitize_filename(filename: str) -> str:
    illegal_chars = r'[<>:"/\\|?*]'
    safe_name = re.sub(illegal_chars, "_", filename)
    if len(safe_name) > 200:
        safe_name = safe_name[:200] + "_truncated"
    return safe_name


def plot_SMILES(SMILES: str, save_dir: str, file_name: str = None, output_format: str = "png") -> dict:
    """
    Plot SMILES molecule as PNG or SVG

    Args:
        SMILES: SMILES string to plot
        save_dir: Directory to save the image
        file_name: Name of the file (default: SMILES string)
        output_format: Output format ("png" or "svg", default: "png")

    Returns:
        Dictionary with success status and file path if successful
    """
    SMILES = Chem.MolToSmiles(Chem.MolFromSmiles(SMILES), kekuleSmiles=True)
    SMILES = SMILES.replace("->", "").replace("<-", "")
    mol = Chem.MolFromSmiles(SMILES)
    file_name = SMILES if file_name is None else file_name
    if mol is None:
        return {"success": False}
    save_path_obj = Path(save_dir)
    if not save_path_obj.exists():
        save_path_obj.mkdir(parents=True, exist_ok=True)

    try:
        indigo = Indigo()
        renderer = IndigoRenderer(indigo)
        safe_name = sanitize_filename(file_name)

        if output_format.lower() == "svg":
            file_path = save_path_obj / f"{safe_name}.svg"
            indigo.setOption("render-output-format", "svg")
        else:
            file_path = save_path_obj / f"{safe_name}.png"
            indigo.setOption("render-output-format", "png")

        mol = indigo.loadMolecule(SMILES)
        mol.layout()
        indigo.setOption("render-coloring", True)
        renderer.renderToFile(mol, str(file_path))

        return {"success": True, "file_path": str(file_path)}
    except Exception as e:
        print(e)
        return {"success": False}
