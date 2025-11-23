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
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
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


def cartesian_product_3d(arr: List[List[Any]], data_type: type, info: str = "") -> np.ndarray:
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
        BarColumn(bar_width=None),
        TimeRemainingColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Computing cartesian product of {info}...", total=num_rows)

        if data_type == object:
            for row_idx, indices in enumerate(cartesian_indices):
                for i, j in enumerate(indices):
                    result[row_idx, i] = arr[i][j]
                progress.update(task, advance=1)
        else:
            for row_idx, indices in enumerate(cartesian_indices):
                col_idx = 0
                for i, j in enumerate(indices):
                    inner_arr = arr[i][j]
                    result[row_idx, col_idx : col_idx + len(inner_arr)] = inner_arr
                    col_idx += len(inner_arr)
                progress.update(task, advance=1)

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
        console.print(f"Normalization error: {str(e)}", style="red")
        console.print("Returning zero array as fallback", style="yellow")
        return np.zeros_like(total_desc_arr)


def array_process(
    desc_dict: Dict[str, pd.DataFrame],
    condition_dict: Dict[str, List[Any]],
    condition_types: List[str],
    desc_normalize: str,
    refine_desc: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Process arrays with rich output.
    Args:
        desc_dict: Descriptor dictionary.
        condition_dict: Condition dictionary.
        condition_types: List of condition type names.
        desc_normalize: Normalization method.
        refine_desc: Method to refine descriptors ('none', 'filter_only', 'auto_select').
    Returns:
        Tuple of (name_array, descriptor_array)
    """
    desc_arrs = [[desc_dict[k].loc[name].values for name in condition_dict[k]] for k in condition_types]
    name_arrs = [list(names) for names in condition_dict.values()]
    # Refine descriptors based on the specified method
    if refine_desc in ["filter_only", "auto_select"]:
        refined_desc_arrs = []
        for desc_arr in desc_arrs:
            if not desc_arr:
                refined_desc_arrs.append([])
                continue
            # --- CORRECTED LOGIC ---
            # Combine all samples of a condition type to calculate inter-descriptor correlation.
            combined_arr = np.vstack(desc_arr)
            df = pd.DataFrame(combined_arr)

            # Calculate correlation matrix and find columns to drop
            corr_matrix = df.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = {column for column in upper.columns if any(upper[column] > 0.7)}

            # Get the indices of columns to keep
            keep_indices = [i for i, col in enumerate(df.columns) if col not in to_drop]

            # Apply filtering to all arrays in the group
            refined_group = df.iloc[:, keep_indices].values
            refined_desc_arrs.append(refined_group)

        desc_arrs = refined_desc_arrs
        # Check total dimensions after filtering
        # Ensure group is not empty before accessing shape
        total_dims = sum(arr.shape[1] for arr in desc_arrs if arr.size > 0)

        # TODO: deal with too many dimension problems
        # if total_dims > 200:
        # console.print(f"[yellow]Warning:[/yellow] Total descriptor dimension after filtering is {total_dims}, which is > 200.")
        # if refine_desc == "auto_select":
        #     console.print("Performing random sampling to reduce dimensions to <= 200...")
        #     # Distribute the 200 dimensions proportionally

        #     current_dims = [arr.shape[1] if arr.size > 0 else 0 for arr in desc_arrs]
        #     target_dims = current_dims.copy()

        #     sampled_desc_arrs = []
        #     for i, desc_arr_group in enumerate(desc_arrs):
        #         if desc_arr_group.size == 0:
        #             sampled_desc_arrs.append([])
        #             continue

        #         num_features = desc_arr_group.shape[1]
        #         k = target_dims[i]
        #         if num_features > k:
        #             indices = np.random.choice(num_features, size=k, replace=False)
        #             indices.sort()  # Keep order for consistency
        #             sampled_group = desc_arr_group[:, indices]
        #             sampled_desc_arrs.append(sampled_group)
        #         else:
        #             sampled_desc_arrs.append(desc_arr_group)
        #     desc_arrs = sampled_desc_arrs

    # Flatten the list of lists for normalization
    # flat_desc_arrs = [item for sublist in desc_arrs for item in sublist]

    # Calculate and print the final total descriptor dimension
    final_total_dims = sum(arr.shape[1] for arr in desc_arrs if arr.size > 0)
    console.print(f"Final total descriptor dimension: [bold cyan]{final_total_dims}[/bold cyan]")
    # Normalize data for each condition *before* cartesian product
    normalized_desc_arrs = []

    for desc_arr in desc_arrs:
        normalized_desc_arrs.append(normalize_data(desc_arr, desc_normalize))

    # Re-group the normalized arrays to match the original structure for cartesian product
    grouped_normalized_arrs = []
    start_index = 0
    for group in desc_arrs:
        end_index = start_index + len(group)
        grouped_normalized_arrs.append(normalized_desc_arrs[start_index:end_index])
        start_index = end_index
    # Perform cartesian product on the processed arrays
    total_desc_arr = cartesian_product_3d(normalized_desc_arrs, data_type=float, info="descriptors")
    total_name_arr = cartesian_product_3d(name_arrs, data_type=object, info="names")
    if len(total_desc_arr) > 0:
        console.print(f"Generated [bold]{len(total_desc_arr):,}[/bold] total combinations", style="green")
    else:
        console.print(
            "[yellow]Warning:[/yellow] No combinations were generated. Check input conditions.",
        )
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
        BarColumn(bar_width=None),
        TimeRemainingColumn(),
        TaskProgressColumn(),
        console=console,
        transient=False,
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
