import numpy as np
import pandas as pd


def get_average_optimal_targets(dfs, target_columns, direction_tags):
    """
    Calculate the average optimal target values across all runs.
    
    For each target, find the optimal value (max for 'max' direction, min for 'min' direction)
    in each DataFrame, then compute the average across all DataFrames.
    
    Args:
        dfs: List of DataFrames containing experimental results
        target_columns: List of target column names
        direction_tags: List of optimization directions ('max' or 'min') for each target
        
    Returns:
        dict: Dictionary mapping target name to its average optimal value
    """
    if len(target_columns) != len(direction_tags):
        raise ValueError("target_columns and direction_tags must have the same length")
    
    results = {}
    
    for col, direction in zip(target_columns, direction_tags):
        optimal_values = []
        
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
        
        avg_optimal = np.mean(optimal_values)
        results[col] = {
            "average_optimal": avg_optimal,
            "all_optimal_values": optimal_values,
            "std": np.std(optimal_values)
        }
    
    return results


def get_auc_of_opt(dfs, target_columns, direction_tags):
    """
    Calculate the Area Under the Curve (AUC) for optimization progress.
    
    For each run (DataFrame):
    1. Calculate cumulative best values at each batch (current and all previous batches)
    2. Calculate the AUC of the cumulative best curve
    
    Then average the AUC values across all runs.
    
    Args:
        dfs: List of DataFrames containing experimental results
        target_columns: List of target column names
        direction_tags: List of optimization directions ('max' or 'min') for each target
        
    Returns:
        dict: Dictionary mapping target name to its average AUC and statistics
    """
    if len(target_columns) != len(direction_tags):
        raise ValueError("target_columns and direction_tags must have the same length")
    
    results = {}
    
    for col, direction in zip(target_columns, direction_tags):
        all_aucs = []
        
        for df in dfs:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")
            
            # Group by batch_index and find the best value for each batch
            batch_best = df.groupby("batch_index")[col].agg('max' if direction == 'max' else 'min')
            
            # Sort by batch index to ensure correct order
            batch_best = batch_best.sort_index()
            
            # Calculate cumulative best (best up to and including current batch)
            if direction == "max":
                cumulative_best = batch_best.cummax()
            else:
                cumulative_best = batch_best.cummin()
            
            # Calculate AUC using trapezoidal rule
            # x values are batch indices (0, 1, 2, ...)
            x = np.arange(len(cumulative_best))
            y = cumulative_best.values
            
            # Calculate AUC
            auc = np.trapz(y, x)
            all_aucs.append(auc)
        
        avg_auc = np.mean(all_aucs)
        results[col] = {
            "average_auc": avg_auc,
            "all_auc_values": all_aucs,
            "std": np.std(all_aucs)
        }
    
    return results


def get_opt_convergence():
    """
    Calculate optimization convergence metrics.
    [Placeholder - to be implemented]
    """
    pass
