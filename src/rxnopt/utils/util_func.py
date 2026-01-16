"""Utility functions for reaction optimization.

Modern utility functions with rich progress bars and improved error handling.
"""

from __future__ import annotations

from functools import wraps


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


def sanitize_filename(filename: str) -> str:
    illegal_chars = r'[<>:"/\\|?*]'
    safe_name = re.sub(illegal_chars, "_", filename)
    if len(safe_name) > 200:
        safe_name = safe_name[:200] + "_truncated"
    return safe_name


def plot_SMILES(SMILES: str, save_dir: str, file_name: str = None) -> dict:
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
        file_path = save_path_obj / f"{safe_name}.png"
        mol = indigo.loadMolecule(SMILES)
        indigo.setOption("render-coloring", True)
        indigo.setOption("render-output-format", "png")
        renderer.renderToFile(mol, str(file_path))

        return {"success": True}
    except Exception as e:
        print(e)
        return {"success": False}
