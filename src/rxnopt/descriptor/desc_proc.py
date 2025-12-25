from itertools import product
from typing import Any, Dict, List, Literal
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
import numpy as np
import pandas as pd
from rich.console import Console

console = Console()


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
        raise Exception("Normalization failed.") from e


def _select_least_correlated_features(df: pd.DataFrame, k: int, feature_type: str = "") -> List[int]:
    """
    从DataFrame中贪心选择k个相关性最低的特征，并显示进度条。
    Args:
        df (pd.DataFrame): 输入的特征DataFrame。
        k (int): 要选择的特征数量。
    Returns:
        List[int]: 被选中的列索引列表。
    """
    num_features = df.shape[1]
    if k >= num_features:
        return list(range(num_features))

    # 如果k为0或负数，直接返回空列表
    if k <= 0:
        return []
    corr_matrix = df.corr().abs()
    # 初始选择第一个特征
    selected_indices = [0]
    candidate_indices = list(range(1, num_features))
    # 设置 rich 进度条
    progress_columns = [
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed} of {task.total})"),
        TimeRemainingColumn(),
    ]
    with Progress(*progress_columns, transient=True) as progress:
        # transient=True 可以在任务完成后移除进度条，使输出更整洁
        task = progress.add_task(f"[cyan]Selecting features of {feature_type}...", total=k)
        # 更新初始选择的第一个特征
        progress.update(task, advance=1)
        while len(selected_indices) < k:
            best_candidate = -1
            lowest_avg_corr = float("inf")

            # 遍历所有候选特征
            for candidate_idx in candidate_indices:
                # 计算候选特征与已选特征集合的平均相关性
                avg_corr = corr_matrix.iloc[candidate_idx, selected_indices].mean()
                if avg_corr < lowest_avg_corr:
                    lowest_avg_corr = avg_corr
                    best_candidate = candidate_idx
            if best_candidate != -1:
                selected_indices.append(best_candidate)
                candidate_indices.remove(best_candidate)
                # 每选择一个新特征，更新进度条
                progress.update(task, advance=1)
            else:
                # 如果没有候选者了（理论上不应发生，除非k>num_features），提前退出
                break
    return selected_indices


def array_process(
    desc_dict: Dict[str, pd.DataFrame],
    condition_dict: Dict[str, List[Any]],
    condition_types: List[str],
    desc_normalize: str,
    refine_desc: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    处理数组，包括描述符筛选、归一化和笛卡尔积。
    Args:
        desc_dict: 描述符字典，键为条件类型，值为包含描述符的DataFrame。
        condition_dict: 条件字典，键为条件类型，值为该类型下的样本名称列表。
        condition_types: 条件类型名称的列表。
        desc_normalize: 归一化方法 ('none', 'standard', etc.)。
        refine_desc: 描述符筛选方法 ('none', 'auto_select', 'filter_0.9', etc.)。
    Returns:
        Tuple[np.ndarray, np.ndarray]: (name_array, descriptor_array)
    """
    # 1. 准备原始数据
    desc_arrs = []
    for k in condition_types:
        # 确保 condition_dict[k] 中的名称存在于 desc_dict[k] 的索引中
        valid_names = [name for name in condition_dict.get(k, []) if name in desc_dict[k].index]
        if valid_names:
            desc_arrs.append(desc_dict[k].loc[valid_names].values)
        else:
            desc_arrs.append(np.array([[]]))  # 添加一个空数组占位
    name_arrs = [list(names) for names in condition_dict.values()]
    # 2. 描述符筛选 (Refine descriptors)
    if refine_desc != "pass":
        refined_desc_arrs = []

        # --- auto_select 逻辑 ---
        if refine_desc == "auto_select":
            console.print("Using 'auto_select' to refine descriptors...")
            MAX_TOTAL_DIMS = 200

            # 计算每个数组的原始维度
            original_dims = np.array([arr.shape[1] for arr in desc_arrs if arr.size > 0])
            total_original_dims = sum(original_dims)
            if total_original_dims > MAX_TOTAL_DIMS:  # 按比例分配目标维度
                proportions = original_dims / total_original_dims
                target_dims = (proportions * MAX_TOTAL_DIMS).astype(int)
                target_dims = np.maximum(1, target_dims)
                target_dims = np.minimum(original_dims, target_dims)

                console.print(f"Total dimensions reduced from {total_original_dims} to ~{sum(target_dims)}.")
                # console.print(f"Target dimensions per group: {target_dims.tolist()}")
            else:
                # 如果总维度本来就不多，则无需缩减
                target_dims = original_dims
                console.print(f"Total dimensions ({total_original_dims}) is within the limit. No reduction needed.")
            arr_idx_with_data = 0
            for i, (desc_arr, f_type) in enumerate(zip(desc_arrs, desc_dict.keys())):
                if desc_arr.size == 0:
                    refined_desc_arrs.append(desc_arr)
                    continue
                k = target_dims[arr_idx_with_data]
                df = pd.DataFrame(desc_arr)

                # 选择k个相关性最低的特征
                keep_indices = _select_least_correlated_features(df, k, f_type)

                refined_group = df.iloc[:, keep_indices].values
                refined_desc_arrs.append(refined_group)
                arr_idx_with_data += 1

            desc_arrs = refined_desc_arrs
        # --- filter_x.x 逻辑 ---
        elif refine_desc.startswith("filter_"):
            try:
                threshold = float(refine_desc.split("_")[1])
                console.print(f"Using '{refine_desc}' to filter descriptors with correlation > {threshold}...")
            except (ValueError, IndexError):
                console.print(f"[red]Error:[/red] Invalid filter format '{refine_desc}'. Expected 'filter_x.x'. Skipping refinement.")
                threshold = -1  # 无效阈值
            if threshold > 0:
                for desc_arr in desc_arrs:
                    if desc_arr.size == 0:
                        refined_desc_arrs.append(desc_arr)
                        continue

                    df = pd.DataFrame(desc_arr)
                    corr_matrix = df.corr().abs()
                    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

                    # 找到要删除的列
                    to_drop = {column for column in upper.columns if any(upper[column] > threshold)}

                    # 应用筛选
                    refined_group = df.drop(columns=to_drop).values
                    refined_desc_arrs.append(refined_group)

                desc_arrs = refined_desc_arrs
        elif refine_desc == "none":
            console.print("No descriptor refinement applied.")
        else:
            console.print(f"[red]Error:[/red] Unknown refine_desc option '{refine_desc}'. No refinement applied.")
            raise Exception(f"Unknown refine_desc option: {refine_desc}")

    # 3. 计算并打印最终维度
    final_total_dims = sum(arr.shape[1] for arr in desc_arrs if arr.size > 0)
    console.print(f"Final total descriptor dimension: [bold cyan]{final_total_dims}[/bold cyan]")
    # 4. 归一化 (Normalize data)
    # 注意：归一化应该在每个独立的数组（代表一个条件类型的所有样本）上进行
    normalized_desc_arrs = []
    for desc_arr in desc_arrs:
        # 对每个非空数组应用归一化
        if desc_arr.size > 0:
            normalized_desc_arrs.append(normalize_data(desc_arr, desc_normalize))
        else:
            # 如果数组为空，则保持原样，但在笛卡尔积中可能需要特殊处理
            # 这里的实现是，如果一个条件类型没有样本，笛卡尔积结果将为空
            normalized_desc_arrs.append(np.array([]))
    # 5. 执行笛卡尔积 (Perform cartesian product)
    # cartesian_product_3d 需要能处理描述符向量的拼接
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
