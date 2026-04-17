import pandas as pd
import numpy as np
import math
from decimal import Decimal, getcontext


def calculate_expected_extreme(df, col_name, n=50, extreme_type="max", num_simulations=10000):
    """
    计算从DataFrame某一列中随机抽取n个样本时，最大值或最小值的期望。

    参数:
    df: pandas.DataFrame, 包含数据的完整数据框
    col_name: str, 目标列的列名
    n: int, 抽样的样本量 (默认50)
    extreme_type: str, 'max' 计算最大值期望, 'min' 计算最小值期望
    num_simulations: int, 蒙特卡洛模拟的次数 (默认10000)

    返回:
    tuple: (模拟期望值, 精确期望值)
    """
    # 提取数据，去除空值，并转换为一维数组
    data = df[col_name].dropna().values
    N = len(data)

    if N < n:
        raise ValueError(f"总体数据量 ({N}) 小于抽样样本量 ({n})，无法进行不放回抽样。")
    if extreme_type not in ["max", "min"]:
        raise ValueError("extreme_type 参数必须是 'max' 或 'min'")

    # ==========================================
    # 1. 蒙特卡洛模拟法 (Monte Carlo Simulation)
    # ==========================================
    simulated_extremes = np.zeros(num_simulations)
    for i in range(num_simulations):
        # 不放回随机抽样
        sample = np.random.choice(data, size=n, replace=False)
        if extreme_type == "max":
            simulated_extremes[i] = np.max(sample)
        else:
            simulated_extremes[i] = np.min(sample)

    sim_expected_value = np.mean(simulated_extremes)

    # ==========================================
    # 2. 精确求解法 (Exact Analytical Method)
    # ==========================================
    # 将数据从小到大排序
    sorted_data = np.sort(data)

    # 计算总的组合数 C(N, n)
    total_combinations = math.comb(N, n)

    # 使用 Python 的超大整数进行累加，避免浮点数精度丢失
    numerator = 0

    if extreme_type == "max":
        # 若排序后的第 i 个元素作为样本的最大值
        # 必须选中该元素，且剩下的 n-1 个元素必须从比它小的 i 个元素中选
        # 因此组合数为 C(i, n-1)
        # 有效索引从 n-1 开始，一直到 N-1
        for i in range(n - 1, N):
            ways = math.comb(i, n - 1)
            numerator += sorted_data[i] * ways

    elif extreme_type == "min":
        # 若排序后的第 i 个元素作为样本的最小值
        # 必须选中该元素，且剩下的 n-1 个元素必须从比它大的 N-1-i 个元素中选
        # 因此组合数为 C(N-1-i, n-1)
        # 有效索引从 0 开始，一直到 N-n
        for i in range(N - n + 1):
            ways = math.comb(N - 1 - i, n - 1)
            numerator += sorted_data[i] * ways

    # 为了防止 N 极大时除法溢出，使用 Decimal 模块进行高精度除法
    getcontext().prec = 50
    exact_expected_value = float(Decimal(numerator) / Decimal(total_combinations))

    return sim_expected_value, exact_expected_value


# ==========================================
# 测试与示例代码
# ==========================================
if __name__ == "__main__":
    # 1. 生成一个模拟数据框 (1000个服从正态分布的样本)
    np.random.seed(42)
    df_mock = pd.read_csv("B-H_HTE.csv")
    target = "cost"

    sample_size = 50

    # 2. 计算最大值的期望
    print(f"--- 抽取 {sample_size} 个样本的最大值期望 ---")
    sim_max, exact_max = calculate_expected_extreme(df_mock, target, n=sample_size, extreme_type="max", num_simulations=10000)
    print(f"模拟解: {sim_max:.4f}")
    print(f"精确解: {exact_max:.4f}")
    print(f"误差率: {abs(sim_max - exact_max) / exact_max * 100:.4f}%\n")

    # 3. 计算最小值的期望
    print(f"--- 抽取 {sample_size} 个样本的最小值期望 ---")
    sim_min, exact_min = calculate_expected_extreme(df_mock, target, n=sample_size, extreme_type="min", num_simulations=10000)
    print(f"模拟解: {sim_min:.4f}")
    print(f"精确解: {exact_min:.4f}")
    print(f"误差率: {abs(sim_min - exact_min) / abs(exact_min) * 100:.4f}%")
