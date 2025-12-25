"""Utility functions for reaction optimization.

Modern utility functions with rich progress bars and improved error handling.
"""

from __future__ import annotations

from functools import wraps

import pandas as pd
import torch
from rich.console import Console

console = Console()


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
