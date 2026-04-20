"""
Random Selection Method Expected Experiments Calculation

This script simulates a purely Random search method to find conditions with
Conversion > threshold for the suzuki_HTE dataset. For each threshold (85-95),
it runs Monte Carlo simulations to estimate the expected number of experiments needed.

Random Algorithm:
1. Randomly select an untested condition from the entire search space.
2. Evaluate its Conversion.
3. If Conversion > threshold, stop and return the number of experiments.
4. Avoid duplicate testing (Sampling without replacement).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, Optional
from tqdm import tqdm


class RandomSimulator:
    """Simulator for Random Selection optimization method."""

    def __init__(self, dataset_path: str, target_col: str = "Conversion"):
        """Initialize the Random simulator."""
        self.dataset = pd.read_csv(dataset_path)
        self.target_col = target_col

        # 将所有的转化率提取为 numpy 数组，方便进行极速的随机打乱和查询
        self.conversions = self.dataset[self.target_col].values
        self.total_space_size = len(self.conversions)

    def simulate_random_run(self, target_threshold: float, random_seed: Optional[int] = None) -> int:
        """Run a single Random simulation to find condition with Conversion > target_threshold."""
        if random_seed is not None:
            np.random.seed(random_seed)

        # 复制一份转化率数据
        conversions = self.conversions.copy()

        # 将数组随机打乱，等价于“无放回的随机抽样”（避免了重复测试）
        np.random.shuffle(conversions)

        # 寻找打乱后，第一个大于目标阈值的索引
        success_mask = conversions > target_threshold

        # 如果搜索空间中完全不存在大于该阈值的条件，则返回整个搜索空间的大小（耗尽全部实验）
        if not np.any(success_mask):
            return self.total_space_size

        # np.argmax 会返回第一个 True 的索引。因为索引从 0 开始，所以实验次数 = 索引 + 1
        experiment_count = int(np.argmax(success_mask)) + 1

        return experiment_count

    def run_monte_carlo(self, target_threshold: float, n_simulations: int = 1000) -> Dict:
        """Run Monte Carlo simulations for a given target threshold."""
        results = []

        # 使用 tqdm 进度条，leave=False 保持终端整洁
        for i in tqdm(range(n_simulations), desc=f"Threshold {target_threshold: >2}", leave=False, ncols=80):
            n_exp = self.simulate_random_run(target_threshold, random_seed=i)
            results.append(n_exp)

        results = np.array(results)
        return {
            "threshold": target_threshold,
            "n_simulations": n_simulations,
            "mean": float(np.mean(results)),
            "median": float(np.median(results)),
            "std": float(np.std(results)),
            "min": int(np.min(results)),
            "max": int(np.max(results)),
        }


def main():
    """Main function to run Random Expected experiments calculation."""
    dataset_path = "/home/tzz/.cline/worktrees/bdca9/synbo/benchmark/datasets/HTE_datasets/suzuki_HTE/suzuki_HTE.csv"

    # 测试阈值从 85 到 95 (包含95)
    thresholds = list(range(90, 100))
    n_simulations = 1000

    print("Loading dataset and initializing Random simulator...")
    simulator = RandomSimulator(dataset_path)

    print(f"Dataset loaded: {simulator.total_space_size} rows")

    all_results = {}
    for threshold in thresholds:
        print(f"\n{'='*60}")
        print(f"Running Monte Carlo simulations for Conversion > {threshold}")
        print(f"{'='*60}")

        result = simulator.run_monte_carlo(threshold, n_simulations)
        all_results[threshold] = result

        print(f"\nResults for Conversion > {threshold}:")
        print(f"  Mean experiments needed: {result['mean']:.2f} ± {result['std']:.2f}")
        print(f"  Median experiments needed: {result['median']:.2f}")
        print(f"  Min/Max: {result['min']} / {result['max']}")

    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "random_expected_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("All results saved to:", output_file)
    print(f"{'='*60}")

    print("\nSummary Table:")
    print(f"{'Threshold':<12} {'Mean':<10} {'Median':<10} {'Std':<10} {'Min':<8} {'Max':<8}")
    print("-" * 70)
    for threshold in thresholds:
        r = all_results[threshold]
        print(f"{threshold:<12} {r['mean']:<10.2f} {r['median']:<10.2f} {r['std']:<10.2f} {r['min']:<8} {r['max']:<8}")


if __name__ == "__main__":
    main()
