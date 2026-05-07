"""
Virtual experiment simulation for SynBO demo.

Provides a ``virtual_experiment`` function that reads a saved batch-results CSV,
replaces placeholder ``[exp_data]`` values with randomly generated experimental
outcomes, and writes the updated CSV back to disk.

This module is designed to **not** be hard-coded inside the notebook so that the
notebook always reads pre-saved CSV files from disk rather than relying on
in-memory variable mutations.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd


def virtual_experiment(
    csv_path: Union[str, Path],
    seed: Optional[int] = None,
    metrics: Optional[List[str]] = None,
    base_values: Optional[dict] = None,
    noise_range: Optional[dict] = None,
) -> pd.DataFrame:
    """Randomly add experimental results to an already-generated batch CSV file.

    Reads the CSV at *csv_path*, replaces every cell in the *metrics* columns
    whose current value is ``[exp_data]`` (or is unparseable as a float) with a
    random sample drawn from a uniform distribution, and then writes the
    modified DataFrame back to the **same file on disk**.

    Parameters
    ----------
    csv_path : str or Path
        Path to the batch results CSV that was saved by
        :meth:`ReactionOptimizer.save_results`.
    seed : int, optional
        Random seed for reproducibility.  When ``None`` the global NumPy
        random state is used.
    metrics : list of str, optional
        Metric column names to simulate.  Defaults to ``['yield', 'ee']``.
    base_values : dict, optional
        Lower bounds for the uniform distribution, keyed by metric name.
        Defaults to ``{'yield': 30, 'ee': 40}``.
    noise_range : dict, optional
        Width of the uniform distribution (upper = base + width), keyed by
        metric name.  Defaults to ``{'yield': 55, 'ee': 55}``.

    Returns
    -------
    pd.DataFrame
        The DataFrame **as read back from disk** after the update (i.e. the
        saved content).

    Raises
    ------
    FileNotFoundError
        If *csv_path* does not exist.
    ValueError
        If none of the *metrics* columns are found in the CSV.

    Examples
    --------
    >>> df = virtual_experiment("results/batch-0_20260506.csv", seed=42)
    >>> df[["yield", "ee"]].head()
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if metrics is None:
        metrics = ["yield", "ee"]

    if base_values is None:
        base_values = {"yield": 30, "ee": 40}

    if noise_range is None:
        noise_range = {"yield": 55, "ee": 55}

    df = pd.read_csv(csv_path)

    missing = [m for m in metrics if m not in df.columns]
    if missing:
        raise ValueError(
            f"Metric column(s) {missing} not found in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )

    if seed is not None:
        np.random.seed(seed)

    for metric in metrics:
        # Identify rows that still hold the placeholder
        col = df[metric]
        # Convert to numeric; cells that cannot be parsed become NaN
        numeric = pd.to_numeric(col, errors="coerce")
        mask = numeric.isna()
        n_fill = mask.sum()

        if n_fill == 0:
            continue

        base = base_values.get(metric, 30)
        width = noise_range.get(metric, 55)
        df.loc[mask, metric] = np.random.uniform(base, base + width, size=n_fill)

    # Write updated results back to disk -----------------------------------
    df.to_csv(csv_path, index=False)

    # Re-read from disk so callers always get the on-disk state ------------
    return pd.read_csv(csv_path)
