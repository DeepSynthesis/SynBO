from itertools import product
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
import numpy as np
import torch
from tqdm import tqdm
from loguru import logger


def track_called(func):
    def wrapper(self, *args, **kwargs):
        setattr(self, f"_{func.__name__}_called", True)
        return func(self, *args, **kwargs)

    return wrapper


def cartesian_product_3d(arr, data_type):
    cartesian_indices = np.array(list(product(*[range(len(middle)) for middle in arr])))
    # 计算结果矩阵的行数和列数
    num_rows = len(cartesian_indices)
    num_cols = sum(len(sub_arr[0]) for sub_arr in arr) if data_type != object else -1
    # 初始化结果矩阵
    result = np.zeros((num_rows, num_cols), dtype=data_type) if data_type != object else np.zeros((num_rows, len(arr)), dtype=data_type)

    # 填充结果矩阵
    if data_type == object:
        for row_idx, indices in tqdm(enumerate(cartesian_indices), total=num_rows):
            for i, j in enumerate(indices):
                result[row_idx, i] = arr[i][j]
    else:
        for row_idx, indices in tqdm(enumerate(cartesian_indices), total=num_rows):
            col_idx = 0
            for i, j in enumerate(indices):
                inner_arr = arr[i][j]
                result[row_idx, col_idx : col_idx + len(inner_arr)] = inner_arr
                col_idx += len(inner_arr)

    return result


def normalize_data(total_desc_arr, desc_normalize):
    """
    Normalize the input array column-wise according to the specified method.

    Parameters:
    total_desc_arr (numpy.ndarray): Array to be normalized (2D array)
    desc_normalize (str): Normalization method, options:
        - 'minmax': Min-max scaling to [0,1] range
        - 'zscore': Standardization (zero mean, unit variance)
        - 'l2': L2 normalization (unit norm)
        - 'none': No normalization

    Returns:
    numpy.ndarray: Normalized array

    Raises:
    ValueError: If unknown normalization method is specified
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
        # Handle cases where normalization might fail (e.g., constant columns)
        print(f"Normalization warning: {str(e)}")
        return np.zeros_like(total_desc_arr)


def array_process(desc_dict, condition_dict, condition_types, desc_normalize):
    desc_arrs = [[desc_dict[k].loc[name].values for name in condition_dict[k]] for k in condition_types]
    name_arrs = [names for names in condition_dict.values()]
    for tp, desc_arr in zip(condition_types, desc_arrs):
        logger.info(f"number of {tp}: {len(desc_arr)}")

    total_desc_arr = cartesian_product_3d(desc_arrs, data_type=float)
    total_name_arr = cartesian_product_3d(name_arrs, data_type=object)

    total_desc_arr = normalize_data(total_desc_arr, desc_normalize)
    return total_name_arr, total_desc_arr


def done_array_process(prev_rxn_info, total_name_arr, condition_types):
    prev_rxn_list = prev_rxn_info[condition_types].to_numpy()
    matches = [np.argwhere(np.all(total_name_arr == row, axis=1)).flatten() for row in prev_rxn_list]
    assert all(len(matches[i]) == 1 for i in range(len(matches))), "Multiple matches found for some reactions"
    assert len(matches) == len(prev_rxn_list), "Number of matches does not match number of reactions"
    matches = np.array(matches).squeeze()
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
