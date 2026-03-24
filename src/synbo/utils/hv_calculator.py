"""Hypervolume calculation utilities for reaction optimization."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


def calculate_hypervolume_for_batch(
    prev_rxn_info: pd.DataFrame,
    opt_metrics: List[str],
    opt_metric_settings: List[Dict[str, any]],
    batch_id: Optional[int] = None,
    reference_point_multiplier: float = 1.0,
) -> Dict[str, any]:
    """
    Calculate hypervolume (HV) for reaction optimization results.

    This function calculates the hypervolume metric for multi-objective optimization,
    which measures the volume of the objective space dominated by the Pareto front.

    Args:
        prev_rxn_info: DataFrame containing previous reaction data with columns:
                       - opt_metrics columns (target values)
                       - 'batch' column (batch index for each reaction)
        opt_metrics: List of optimization metric column names
        opt_metric_settings: List of dictionaries with settings for each metric:
                            - 'opt_direct': 'max' or 'min'
                            - 'opt_range': [min_value, max_value]
                            - 'metric_weight': weight for multi-objective (optional)
        batch_id: Optional batch ID to calculate HV up to. If None, uses all data
        reference_point_multiplier: Multiplier for reference point (default: 1.0)

    Returns:
        Dictionary containing:
            - 'hv': Hypervolume value
            - 'hv_normalized': Normalized hypervolume (0 to 1)
            - 'total_hv': Total theoretical hypervolume (reference)
            - 'num_points': Number of points used in calculation
            - 'batch_id': Batch ID used for calculation
    """
    from pymoo.indicators.hv import HV

    # Validate inputs
    if len(opt_metrics) != len(opt_metric_settings):
        raise ValueError("opt_metrics and opt_metric_settings must have same length")

    for metric in opt_metrics:
        if metric not in prev_rxn_info.columns:
            raise ValueError(f"Metric '{metric}' not found in prev_rxn_info")

    if "batch" not in prev_rxn_info.columns:
        raise ValueError("'batch' column not found in prev_rxn_info")

    # Filter data up to specified batch if provided
    if batch_id is not None:
        df_filtered = prev_rxn_info[prev_rxn_info["batch"] <= batch_id].copy()
    else:
        df_filtered = prev_rxn_info.copy()

    if len(df_filtered) == 0:
        raise ValueError(f"No data found for batch_id {batch_id}")

    # Extract raw metric values
    data_raw = df_filtered[opt_metrics].values

    def normalize_and_transform(data, metric_settings):
        """
        Normalize data to [0, 1] using opt_range and convert to minimization problem.

        For minimization: keep normalized value as is
        For maximization: transform to 1 - normalized value
        """
        data = np.array(data, dtype=float)
        transformed = np.zeros_like(data)

        for i, setting in enumerate(metric_settings):
            col_data = data[:, i]
            opt_range = setting["opt_range"]
            direction = setting["opt_direct"]

            col_min, col_max = opt_range

            # Normalize to [0, 1]
            if col_max == col_min:
                norm_col = np.full_like(col_data, 1.0)
            else:
                norm_col = (col_data - col_min) / (col_max - col_min)

            # Clip to [0, 1]
            norm_col = np.clip(norm_col, 0.0, 1.0)

            # Convert to minimization problem
            if direction == "min":
                transformed[:, i] = norm_col
            elif direction == "max":
                transformed[:, i] = 1.0 - norm_col
            else:
                raise ValueError(f"Unknown opt_direct '{direction}'. Use 'max' or 'min'.")

        return transformed

    # Transform data for hypervolume calculation
    data_norm = normalize_and_transform(data_raw, opt_metric_settings)

    # Set reference point (slightly outside the normalized space [0, 1])
    ref_point = np.array([reference_point_multiplier] * len(opt_metrics))

    # Calculate hypervolume
    hv_calculator = HV(ref_point=ref_point)
    hv_value = hv_calculator(data_norm)

    # Calculate total theoretical hypervolume (maximum possible)
    # This is when all objectives are at their optimal values (0 for minimization)
    # The reference point defines the bounding box
    total_hv = np.prod(ref_point)

    # Normalize hypervolume (0 to 1)
    hv_normalized = hv_value / total_hv if total_hv > 0 else 0.0

    # Prepare result
    result = {
        "hv": float(hv_value),
        "hv_normalized": float(hv_normalized),
        "total_hv": float(total_hv),
        "num_points": len(df_filtered),
        "batch_id": batch_id if batch_id is not None else df_filtered["batch"].max(),
    }

    return result


def calculate_hypervolume_by_batch(
    prev_rxn_info: pd.DataFrame,
    opt_metrics: List[str],
    opt_metric_settings: List[Dict[str, any]],
    reference_point_multiplier: float = 1.0,
) -> pd.DataFrame:
    """
    Calculate hypervolume for each batch cumulatively.

    This function calculates the hypervolume at each batch, including all data
    from previous batches. This shows the progress of optimization over time.

    Args:
        prev_rxn_info: DataFrame containing previous reaction data with columns:
                       - opt_metrics columns (target values)
                       - 'batch' column (batch index for each reaction)
        opt_metrics: List of optimization metric column names
        opt_metric_settings: List of dictionaries with settings for each metric
        reference_point_multiplier: Multiplier for reference point (default: 1.0)

    Returns:
        DataFrame with columns:
            - 'batch': Batch index
            - 'hv': Hypervolume value
            - 'hv_normalized': Normalized hypervolume (0 to 1)
            - 'num_points': Cumulative number of points
    """
    # Get all unique batch IDs
    batch_ids = sorted(prev_rxn_info["batch"].unique())

    results = []

    for batch_id in batch_ids:
        try:
            hv_result = calculate_hypervolume_for_batch(
                prev_rxn_info=prev_rxn_info,
                opt_metrics=opt_metrics,
                opt_metric_settings=opt_metric_settings,
                batch_id=batch_id,
                reference_point_multiplier=reference_point_multiplier,
            )

            results.append(
                {
                    "batch": hv_result["batch_id"],
                    "hv": hv_result["hv"],
                    "hv_normalized": hv_result["hv_normalized"],
                    "num_points": hv_result["num_points"],
                }
            )
        except Exception as e:
            print(f"Warning: Could not calculate HV for batch {batch_id}: {e}")
            continue

    return pd.DataFrame(results)
