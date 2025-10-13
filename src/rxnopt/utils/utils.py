"""Utility functions for reaction optimization.

Modern utility functions with rich progress bars and improved error handling.
"""

from __future__ import annotations

from functools import wraps
from itertools import product
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer

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


def cartesian_product_3d(arr: List[List[Any]], data_type: type) -> np.ndarray:
    """Create cartesian product of 3D array with rich progress bar.

    Args:
        arr: List of lists containing arrays to combine
        data_type: Data type for output array

    Returns:
        NumPy array containing cartesian product
    """
    cartesian_indices = np.array(list(product(*[range(len(middle)) for middle in arr])))
    num_rows = len(cartesian_indices)

    if data_type == object:
        result = np.zeros((num_rows, len(arr)), dtype=data_type)
    else:
        num_cols = sum(len(sub_arr[0]) for sub_arr in arr)
        result = np.zeros((num_rows, num_cols), dtype=data_type)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Computing cartesian product...", total=num_rows)

        if data_type == object:
            for row_idx, indices in enumerate(cartesian_indices):
                for i, j in enumerate(indices):
                    result[row_idx, i] = arr[i][j]
                if row_idx % 1000 == 0:  # Update progress every 1000 iterations
                    progress.update(task, completed=row_idx)
        else:
            for row_idx, indices in enumerate(cartesian_indices):
                col_idx = 0
                for i, j in enumerate(indices):
                    inner_arr = arr[i][j]
                    result[row_idx, col_idx : col_idx + len(inner_arr)] = inner_arr
                    col_idx += len(inner_arr)
                if row_idx % 1000 == 0:
                    progress.update(task, completed=row_idx)

        progress.update(task, completed=num_rows)

    return result


def normalize_data(total_desc_arr: np.ndarray, desc_normalize: Literal["minmax", "zscore", "l2", "none"]) -> np.ndarray:
    """Normalize array with modern error handling.

    Args:
        total_desc_arr: Array to normalize (2D)
        desc_normalize: Normalization method

    Returns:
        Normalized array

    Raises:
        ValueError: If unknown normalization method specified
    """
    try:
        match desc_normalize:
            case "minmax":
                return MinMaxScaler().fit_transform(total_desc_arr)
            case "zscore":
                return StandardScaler().fit_transform(total_desc_arr)
            case "l2":
                return Normalizer(norm="l2").fit_transform(total_desc_arr)
            case "none":
                return total_desc_arr.copy()
            case _:
                raise ValueError(f"Unknown normalization method: {desc_normalize}")
    except ValueError as e:
        console.print(f"Normalization error: {str(e)}", style='red')
        console.print("Returning zero array as fallback", style='yellow')
        return np.zeros_like(total_desc_arr)


def array_process(
    desc_dict: Dict[str, Any], condition_dict: Dict[str, List[Any]], condition_types: List[str], desc_normalize: str
) -> tuple[np.ndarray, np.ndarray]:
    """Process arrays with rich output.

    Args:
        desc_dict: Descriptor dictionary
        condition_dict: Condition dictionary
        condition_types: List of condition type names
        desc_normalize: Normalization method

    Returns:
        Tuple of (name_array, descriptor_array)
    """
    desc_arrs = [[desc_dict[k].loc[name].values for name in condition_dict[k]] for k in condition_types]
    name_arrs = [list(names) for names in condition_dict.values()]

    # # Display condition counts
    # for condition_type, desc_arr in zip(condition_types, desc_arrs):
    #     console.print(f"[cyan]{condition_type}[/cyan]: {len(desc_arr)} conditions")

    total_desc_arr = cartesian_product_3d(desc_arrs, data_type=float)
    total_name_arr = cartesian_product_3d(name_arrs, data_type=object)

    console.print(f"Generated [bold]{len(total_desc_arr):,}[/bold] total combinations", style="green")

    with Progress(
        SpinnerColumn(),
        TextColumn("Normalizing descriptors..."),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("normalizing", total=None)
        total_desc_arr = normalize_data(total_desc_arr, desc_normalize)

    return total_name_arr, total_desc_arr


def done_array_process(prev_rxn_info: pd.DataFrame, total_name_arr: np.ndarray, condition_types: List[str]) -> np.ndarray:
    """Process completed reactions with validation.

    Args:
        prev_rxn_info: Previous reaction information
        total_name_arr: Array of all condition combinations
        condition_types: List of condition type names

    Returns:
        Indices of completed reactions in total array

    Raises:
        AssertionError: If multiple matches found or counts don't match
    """
    prev_rxn_list = prev_rxn_info[condition_types].to_numpy()

    with Progress(
        SpinnerColumn(),
        TextColumn("Matching completed reactions..."),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("matching", total=len(prev_rxn_list))

        matches = []
        for i, row in enumerate(prev_rxn_list):
            match_indices = np.argwhere(np.all(total_name_arr == row, axis=1)).flatten()
            matches.append(match_indices)
            progress.update(task, completed=i + 1)

    # Validation
    invalid_matches = [i for i, match in enumerate(matches) if len(match) != 1]
    if invalid_matches:
        raise AssertionError(f"Multiple or no matches found for reactions: {invalid_matches}")

    if len(matches) != len(prev_rxn_list):
        raise AssertionError(f"Number of matches ({len(matches)}) does not match " f"number of reactions ({len(prev_rxn_list)})")

    matches = np.array(matches).squeeze()
    console.print(f"Matched {len(matches)} completed reactions", style="green")

    return matches


def generate_onehot_desc(condition_dict):
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
