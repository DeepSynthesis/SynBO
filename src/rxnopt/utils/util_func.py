"""Utility functions for reaction optimization.

Modern utility functions with rich progress bars and improved error handling.
"""

from __future__ import annotations

from functools import wraps


import pandas as pd
import torch
from pathlib import Path
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
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


def plot_SMILES(SMILES: str, save_dir: str) -> dict:
    mol = Chem.MolFromSmiles(SMILES)
    if mol is None:
        return {"success": False}
    save_path_obj = Path(save_dir)
    if not save_path_obj.exists():
        save_path_obj.mkdir(parents=True, exist_ok=True)
    width, height = 400, 400
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    opts = drawer.drawOptions()
    # --- 样式设置 ---
    opts.bondLineWidth = 3.0  # 键的粗细
    opts.minFontSize = 16  # 最小字体大小
    opts.fixedFontSize = 20  # 固定字体大小
    # opts.atomLabelFontFace = "Arial"  <-- 删除这一行，RDKit不支持此属性

    # RDKit 默认就是类似 Arial 的无衬线字体
    try:
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        # 确保这里调用了正确的文件名清洗函数
        # safe_name = sanitize_filename(SMILES)
        # 临时替代方案，防止 sanitize_filename 未定义导致测试失败：
        safe_name = sanitize_filename(SMILES)
        file_path = save_path_obj / f"{safe_name}.png"
        drawer.WriteDrawingText(str(file_path))
        return {"success": True}
    except Exception as e:
        return {"success": False}


