from pathlib import Path
import numpy as np
import pandas as pd


def get_average_optimal_targets(dfs, target_columns, direction_tags, range_tags):
    """
    Calculate average optimal target values across all runs with normalization.

    For each target, find optimal value (max for 'max' direction, min for 'min' direction)
    in each DataFrame, normalize value to [0,1] range, and compute average.
    For minimization targets, normalized value is flipped so all targets are "higher is better".

    Args:
        dfs: List of DataFrames containing experimental results
        target_columns: List of target column names
        direction_tags: List of optimization directions ('max' or 'min') for each target
        range_tags: List of [min, max] ranges for each target for normalization

    Returns:
        dict: Dictionary mapping target name to its average normalized optimal value
    """
    if len(target_columns) != len(direction_tags) or len(target_columns) != len(range_tags):
        raise ValueError("target_columns, direction_tags, and range_tags must have same length")

    results = {}

    for col, direction, (min_val, max_val) in zip(target_columns, direction_tags, range_tags):
        optimal_values = []
        normalized_values = []

        for df in dfs:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")

            if direction == "max":
                optimal_value = df[col].max()
            elif direction == "min":
                optimal_value = df[col].min()
            else:
                raise ValueError(f"Unknown direction '{direction}' for column '{col}'. Use 'max' or 'min'.")

            optimal_values.append(optimal_value)

            # Normalize to [0, 1] and flip if necessary
            if max_val == min_val:
                normalized = 0.0
            else:
                if direction == "max":
                    # Maximize: (value - min) / (max - min) -> higher is better
                    normalized = (optimal_value - min_val) / (max_val - min_val)
                else:
                    # Minimize: (max - value) / (max - min) -> higher is better (flipped)
                    normalized = (max_val - optimal_value) / (max_val - min_val)

            normalized = np.clip(normalized, 0.0, 1.0)
            normalized_values.append(normalized)

        avg_optimal = np.mean(optimal_values)
        avg_normalized = np.mean(normalized_values)
        results[col] = {
            "average_optimal": avg_optimal,
            "average_normalized": avg_normalized,
            "std": np.std(optimal_values),
            "std_normalized": np.std(normalized_values),
        }

    return results


def get_auc_of_opt(dfs, target_columns, direction_tags, range_tags):
    """
    Calculate Area Under the Curve (AUC) for optimization progress with normalization.

    For each run (DataFrame):
    1. Calculate cumulative best values at each batch (current and all previous batches)
    2. Calculate AUC of the cumulative best curve
    3. Normalize AUC by multiplying with number of batches

    For minimization targets, AUC is calculated on flipped values so all targets are "higher is better".

    Then average the AUC values across all runs.

    Args:
        dfs: List of DataFrames containing experimental results
        target_columns: List of target column names
        direction_tags: List of optimization directions ('max' or 'min') for each target
        range_tags: List of [min, max] ranges for each target for normalization

    Returns:
        dict: Dictionary mapping target name to its average normalized AUC and statistics
    """
    if len(target_columns) != len(direction_tags) or len(target_columns) != len(range_tags):
        raise ValueError("target_columns, direction_tags, and range_tags must have same length")

    results = {}

    for col, direction, (min_val, max_val) in zip(target_columns, direction_tags, range_tags):
        all_aucs = []
        all_normalized_aucs = []

        for df in dfs:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")

            # Group by batch_index and find the best value for each batch
            batch_best = df.groupby("batch_index")[col].agg("max" if direction == "max" else "min")

            # Sort by batch index to ensure correct order
            batch_best = batch_best.sort_index()

            # Calculate cumulative best (best up to and including current batch)
            if direction == "max":
                cumulative_best = batch_best.cummax()
            else:
                cumulative_best = batch_best.cummin()

            # For minimization, flip the values
            if direction == "min":
                cumulative_best = max_val - cumulative_best
                min_val, max_val = 0, max_val - min_val

            # Calculate AUC using trapezoidal rule
            # x values are batch indices (0, 1, 2, ...)
            x = np.arange(len(cumulative_best))
            y = cumulative_best.values

            # Calculate AUC
            auc = np.trapezoid(y, x=x)
            all_aucs.append(auc)

            # Normalize AUC by multiplying with number of batches
            # Maximum possible AUC is max_value * num_batches
            num_batches = len(cumulative_best)
            if num_batches > 0:
                normalized_auc = auc / (max_val * num_batches) if max_val != 0 else 0.0
            else:
                normalized_auc = 0.0

            normalized_auc = np.clip(normalized_auc, 0.0, 1.0)
            all_normalized_aucs.append(normalized_auc)

        avg_auc = np.mean(all_aucs)
        avg_normalized_auc = np.mean(all_normalized_aucs)
        results[col] = {
            "average": avg_auc,
            "average_normalized": avg_normalized_auc,
            "std": np.std(all_aucs),
            "std_normalized": np.std(all_normalized_aucs),
        }

    return results


def get_hypervolume(dfs, target_columns, direction_tags, range_tags, full_space_file):
    """
    Calculate hypervolume (HV) metrics for optimization results.

    Similar to plot_hypervolume_coverage logic:
    1. Normalize all data to [0, 1] using range_tags
    2. Convert all objectives to minimization (pymoo HV default)
    3. Calculate HV for each run at final batch
    4. Calculate HV coverage percentage relative to full space HV

    Args:
        dfs: List of DataFrames containing experimental results with target_columns and 'batch_index'
        target_columns: List of target column names
        direction_tags: Optimization direction list, e.g., ['max', 'min', 'max']
        range_tags: Theoretical range list, List[tuple(min, max)], for normalization
        full_space_file: Full space data file path (CSV or Excel)

    Returns:
        dict: Dictionary with HV metrics including:
            - average_hv: Average final HV across all runs
            - average_hv_coverage: Average HV coverage percentage
            - std: Standard deviation of final HV
            - std_coverage: Standard deviation of HV coverage percentage
    """
    from pymoo.indicators.hv import HV

    if len(target_columns) != len(direction_tags) or len(target_columns) != len(range_tags):
        raise ValueError("target_columns, direction_tags, and range_tags must have same length")

    # Load full space data
    full_space_file = Path(full_space_file)
    if str(full_space_file).endswith(".csv"):
        full_df = pd.read_csv(full_space_file)
    else:
        full_df = pd.read_excel(full_space_file)

    full_data_raw = full_df[target_columns].values

    def normalize_and_transform(data, ranges, directions):
        """
        1. Normalize to [0, 1] using range_tags
        2. Convert to minimization problem using direction_tags
        """
        data = np.array(data)
        transformed = np.zeros_like(data, dtype=float)

        for i, (col_min, col_max) in enumerate(ranges):
            col_data = data[:, i]
            if col_max == col_min:
                norm_col = np.full_like(col_data, 1.0)
            else:
                norm_col = (col_data - col_min) / (col_max - col_min)

            norm_col = np.clip(norm_col, 0.0, 1.0)

            direction = directions[i]
            if direction == "min":
                transformed[:, i] = norm_col
            elif direction == "max":
                transformed[:, i] = 1.0 - norm_col
            else:
                print(f"Warning: Unknown direction '{direction}', assuming 'max'.")
                transformed[:, i] = 1.0 - norm_col

        return transformed

    # Calculate total theoretical HV from full space
    full_data_norm = normalize_and_transform(full_data_raw, range_tags, direction_tags)
    ref_point = np.array([1.1] * len(target_columns))
    ind = HV(ref_point=ref_point)
    total_hv = ind(full_data_norm)

    if total_hv == 0:
        print("Warning: Total Hypervolume is 0. Check your range_tags or data.")
        return {"average_hv": 0.0, "average_hv_coverage": 0.0, "std": 0.0, "std_coverage": 0.0, "total_hv": 0.0}

    # Calculate HV for each run
    all_final_hvs = []
    all_hv_coverages = []

    for df in dfs:
        df = df.sort_values("batch_index").copy()

        # Get all samples (cumulative up to final batch)
        round_data_raw = df[target_columns].values
        round_data_norm = normalize_and_transform(round_data_raw, range_tags, direction_tags)

        # Calculate final HV
        final_hv = ind(round_data_norm)
        hv_coverage = (final_hv / total_hv) * 100

        all_final_hvs.append(final_hv)
        all_hv_coverages.append(hv_coverage)

    # Calculate statistics
    avg_hv = np.mean(all_final_hvs)
    avg_hv_coverage = np.mean(all_hv_coverages)

    results = {
        "average_hv": avg_hv,
        "average_hv_coverage": avg_hv_coverage,
        "std": np.std(all_final_hvs),
        "std_coverage": np.std(all_hv_coverages),
        "total_hv": total_hv,
    }

    return results


def get_average_optimal_targets_hv(dfs, target_columns, direction_tags, range_tags, full_space_file):
    """
    Calculate average optimal hypervolume values across all runs with normalization.

    For each run, calculate the final hypervolume value and normalize it by total_hv.
    This gives a normalized HV metric where 1.0 means the run achieved full space HV.

    Args:
        dfs: List of DataFrames containing experimental results with target_columns and 'batch_index'
        target_columns: List of target column names
        direction_tags: Optimization direction list, e.g., ['max', 'min', 'max']
        range_tags: Theoretical range list, List[tuple(min, max)], for normalization
        full_space_file: Full space data file path (CSV or Excel)

    Returns:
        dict: Dictionary with normalized HV metrics:
            - average_hv_normalized: Average normalized final HV (0 to 1)
            - std: Standard deviation of normalized final HV
            - total_hv: Total theoretical HV from full space
    """
    from pymoo.indicators.hv import HV

    if len(target_columns) != len(direction_tags) or len(target_columns) != len(range_tags):
        raise ValueError("target_columns, direction_tags, and range_tags must have same length")

    # Load full space data
    full_space_file = Path(full_space_file)
    if str(full_space_file).endswith(".csv"):
        full_df = pd.read_csv(full_space_file)
    else:
        full_df = pd.read_excel(full_space_file)

    full_data_raw = full_df[target_columns].values

    def normalize_and_transform(data, ranges, directions):
        """
        1. Normalize to [0, 1] using range_tags
        2. Convert to minimization problem using direction_tags
        """
        data = np.array(data)
        transformed = np.zeros_like(data, dtype=float)

        for i, (col_min, col_max) in enumerate(ranges):
            col_data = data[:, i]
            if col_max == col_min:
                norm_col = np.full_like(col_data, 1.0)
            else:
                norm_col = (col_data - col_min) / (col_max - col_min)

            norm_col = np.clip(norm_col, 0.0, 1.0)

            direction = directions[i]
            if direction == "min":
                transformed[:, i] = norm_col
            elif direction == "max":
                transformed[:, i] = 1.0 - norm_col
            else:
                print(f"Warning: Unknown direction '{direction}', assuming 'max'.")
                transformed[:, i] = 1.0 - norm_col

        return transformed

    # Calculate total theoretical HV from full space
    full_data_norm = normalize_and_transform(full_data_raw, range_tags, direction_tags)
    ref_point = np.array([1.1] * len(target_columns))
    ind = HV(ref_point=ref_point)
    # total_hv = ind(full_data_norm)
    total_hv = 0

    if total_hv == 0:
        print("Warning: Total Hypervolume is 0. Check your range_tags or data.")
        return {
            "average_normalized": 0.0,
            "std_normalized": 0.0,
            "total_hv": 0.0,
        }

    # Calculate normalized HV for each run
    all_normalized_hvs = []

    for df in dfs:
        df = df.sort_values("batch_index").copy()

        # Get all samples (cumulative up to final batch)
        round_data_raw = df[target_columns].values
        round_data_norm = normalize_and_transform(round_data_raw, range_tags, direction_tags)

        # Calculate final HV and normalize by total_hv
        final_hv = ind(round_data_norm)
        normalized_hv = final_hv / total_hv

        all_normalized_hvs.append(normalized_hv)

    # Calculate statistics
    avg_hv_normalized = np.mean(all_normalized_hvs)

    results = {"average_normalized": avg_hv_normalized, "std_normalized": np.std(all_normalized_hvs), "total_hv": total_hv}

    return results


def get_auc_of_opt_hv(dfs, target_columns, direction_tags, range_tags, full_space_file):
    """
    Calculate Area Under the Curve (AUC) for hypervolume optimization progress with normalization.

    For each run (DataFrame):
    1. Calculate HV at each batch (current and all previous batches)
    2. Calculate AUC of the HV curve
    3. Normalize AUC by total_hv * num_batches

    Then average the AUC values across all runs.

    Args:
        dfs: List of DataFrames containing experimental results with target_columns and 'batch_index'
        target_columns: List of target column names
        direction_tags: Optimization direction list, e.g., ['max', 'min', 'max']
        range_tags: Theoretical range list, List[tuple(min, max)], for normalization
        full_space_file: Full space data file path (CSV or Excel)

    Returns:
        dict: Dictionary with HV AUC metrics:
            - average_auc_hv: Average HV AUC across all runs
            - average_auc_hv_normalized: Average normalized HV AUC (0 to 1)
            - std: Standard deviation of HV AUC
            - std_normalized: Standard deviation of normalized HV AUC
            - total_hv: Total theoretical HV from full space
    """
    from pymoo.indicators.hv import HV

    if len(target_columns) != len(direction_tags) or len(target_columns) != len(range_tags):
        raise ValueError("target_columns, direction_tags, and range_tags must have same length")

    # Load full space data
    full_space_file = Path(full_space_file)
    if str(full_space_file).endswith(".csv"):
        full_df = pd.read_csv(full_space_file)
    else:
        full_df = pd.read_excel(full_space_file)

    full_data_raw = full_df[target_columns].values

    def normalize_and_transform(data, ranges, directions):
        """
        1. Normalize to [0, 1] using range_tags
        2. Convert to minimization problem using direction_tags
        """
        data = np.array(data)
        transformed = np.zeros_like(data, dtype=float)

        for i, (col_min, col_max) in enumerate(ranges):
            col_data = data[:, i]
            if col_max == col_min:
                norm_col = np.full_like(col_data, 1.0)
            else:
                norm_col = (col_data - col_min) / (col_max - col_min)

            norm_col = np.clip(norm_col, 0.0, 1.0)

            direction = directions[i]
            if direction == "min":
                transformed[:, i] = norm_col
            elif direction == "max":
                transformed[:, i] = 1.0 - norm_col
            else:
                print(f"Warning: Unknown direction '{direction}', assuming 'max'.")
                transformed[:, i] = 1.0 - norm_col

        return transformed

    # Calculate total theoretical HV from full space
    full_data_norm = normalize_and_transform(full_data_raw, range_tags, direction_tags)
    ref_point = np.array([1.1] * len(target_columns))
    ind = HV(ref_point=ref_point)
    # total_hv = ind(full_data_norm)
    total_hv = 0

    if total_hv == 0:
        print("Warning: Total Hypervolume is 0. Check your range_tags or data.")
        return {"average": 0.0, "average_normalized": 0.0, "std": 0.0, "std_normalized": 0.0, "total_hv": 0.0}

    # Calculate HV AUC for each run
    all_auc_hvs = []
    all_auc_hvs_normalized = []

    for df in dfs:
        df = df.sort_values("batch_index").copy()
        round_data_raw = df[target_columns].values

        # Calculate HV at each batch (cumulative)
        batch_hv_values = {}
        batch_indices = sorted(df["batch_index"].unique())

        for batch_idx in batch_indices:
            # Get all samples up to and including this batch
            batch_samples = df[df["batch_index"] <= batch_idx]
            batch_data_raw = batch_samples[target_columns].values
            batch_data_norm = normalize_and_transform(batch_data_raw, range_tags, direction_tags)

            # Calculate HV for this batch
            current_hv = ind(batch_data_norm)
            batch_hv_values[batch_idx] = current_hv

        # Convert to series for AUC calculation
        batch_hv_series = pd.Series(batch_hv_values).sort_index()

        # Calculate AUC using trapezoidal rule
        # x values are batch indices (0, 1, 2, ...)
        x = np.arange(len(batch_hv_series))
        y = batch_hv_series.values

        # Calculate AUC
        auc_hv = np.trapezoid(y, x=x)
        all_auc_hvs.append(auc_hv)

        # Normalize AUC by total_hv * num_batches
        num_batches = len(batch_hv_series)
        if num_batches > 0 and total_hv > 0:
            normalized_auc_hv = auc_hv / (total_hv * num_batches)
        else:
            normalized_auc_hv = 0.0

        normalized_auc_hv = np.clip(normalized_auc_hv, 0.0, 1.0)
        all_auc_hvs_normalized.append(normalized_auc_hv)

    # Calculate statistics
    avg_auc_hv = np.mean(all_auc_hvs)
    avg_auc_hv_normalized = np.mean(all_auc_hvs_normalized)

    results = {
        "average": avg_auc_hv,
        "average_normalized": avg_auc_hv_normalized,
        "std": np.std(all_auc_hvs),
        "std_normalized": np.std(all_auc_hvs_normalized),
        "total_hv": total_hv,
    }

    return results
