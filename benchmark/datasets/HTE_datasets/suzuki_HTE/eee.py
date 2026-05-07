import pandas as pd
import numpy as np
import math
from decimal import Decimal, getcontext


def calculate_expected_extreme(df, col_name, n=50, extreme_type="max", num_simulations=10000):
    """
    Compute the expected maximum or minimum when randomly drawing n samples from a DataFrame column.

    Parameters:
    df: pandas.DataFrame, full dataset
    col_name: str, target column name
    n: int, sample size (default 50)
    extreme_type: str, 'max' for expected maximum, 'min' for expected minimum
    num_simulations: int, number of Monte Carlo simulations (default 10000)

    Returns:
    tuple: (simulated expected value, exact expected value)
    """
    # Extract data, remove nulls, and convert to 1D array
    data = df[col_name].dropna().values
    N = len(data)

    if N < n:
        raise ValueError(f"Population size ({N}) is smaller than sample size ({n}), cannot sample without replacement.")
    if extreme_type not in ["max", "min"]:
        raise ValueError("extreme_type must be 'max' or 'min'")

    # ==========================================
    # 1. Monte Carlo Simulation
    # ==========================================
    simulated_extremes = np.zeros(num_simulations)
    for i in range(num_simulations):
        # Random sampling without replacement
        sample = np.random.choice(data, size=n, replace=False)
        if extreme_type == "max":
            simulated_extremes[i] = np.max(sample)
        else:
            simulated_extremes[i] = np.min(sample)

    sim_expected_value = np.mean(simulated_extremes)

    return sim_expected_value


# ==========================================
# Test and example code
# ==========================================
if __name__ == "__main__":
    # 1. Generate a simulated DataFrame (1000 normally distributed samples)
    np.random.seed(42)
    df_mock = pd.read_csv("suzuki_HTE.csv")
    target = "Conversion"

    sample_size = 300

    # 2. Compute expected maximum
    print(f"--- Drawing {sample_size}  samples: expected maximum ---")
    sim_max= calculate_expected_extreme(df_mock, target, n=sample_size, extreme_type="max", num_simulations=10000)
    print(f"Simulated: {sim_max:.4f}")


    # 3. Compute expected minimum
    print(f"--- Drawing {sample_size}  samples: expected minimum ---")
    sim_min = calculate_expected_extreme(df_mock, target, n=sample_size, extreme_type="min", num_simulations=10000)
    print(f"Simulated: {sim_min:.4f}")
