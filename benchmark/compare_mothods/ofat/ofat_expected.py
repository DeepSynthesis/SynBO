"""
OFAT (One-Factor-At-A-Time) Method Expected Experiments Calculation

This script simulates the OFAT method to find conditions with Conversion > threshold
for the suzuki_HTE dataset. For each threshold (85-95), it runs Monte Carlo simulations
to estimate the expected number of experiments needed.

OFAT Algorithm:
1. Randomly select a starting reagent_type from the list
2. For other reagent_types, randomly select candidates
3. For the selected reagent_type, test all its candidates to find the optimal one
4. Fix the optimal candidate and move to the next reagent_type
5. If target not reached after all reagent_types, use the best known condition and
   restart from the first reagent_type. If a reagent_type has been fully optimized,
   step down to its second-best (or third-best) to escape local optima.
6. Avoid duplicate testing
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Set, Optional
import random
from tqdm import tqdm


class OFATSimulator:
    """Simulator for One-Factor-At-A-Time (OFAT) optimization method."""

    def __init__(self, dataset_path: str, reagent_types: List[str], target_col: str = "Conversion"):
        """Initialize the OFAT simulator."""
        self.dataset = pd.read_csv(dataset_path)
        self.reagent_types = reagent_types
        self.target_col = target_col

        # 优化提速：记录试剂类型对应的索引，将后续的字典操作转化为更快的列表操作
        self.rt_to_idx = {rt: i for i, rt in enumerate(reagent_types)}

        self.candidates = {}
        for rt in reagent_types:
            self.candidates[rt] = self.dataset[rt].unique().tolist()

        self.total_space_size = len(self.dataset)
        self._build_lookup_table()

    def _build_lookup_table(self):
        """Build a lookup table for quick conversion value retrieval."""
        self.lookup = {}
        for _, row in self.dataset.iterrows():
            key = tuple(row[rt] for rt in self.reagent_types)
            self.lookup[key] = row[self.target_col]

    def get_conversion_by_key(self, key: Tuple) -> float:
        """Get the conversion value for a given condition key."""
        return self.lookup.get(key, 0.0)

    def simulate_ofat_run(self, target_threshold: float, random_seed: Optional[int] = None) -> int:
        """Run a single OFAT simulation to find condition with Conversion > target_threshold."""
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        tested_conditions: Set[Tuple] = set()

        # 1 & 2. 初始化：其他试剂随机选择 candidates
        current_state = [random.choice(self.candidates[rt]) for rt in self.reagent_types]
        best_state = current_state.copy()
        best_conversion = -np.inf

        experiment_count = 0

        # 随机决定遍历试剂的顺序
        order = self.reagent_types.copy()
        random.shuffle(order)

        # 记录每个试剂当前选用的排名索引 (0代表最优, 1代表次优, ...)
        rank_target = {rt: 0 for rt in self.reagent_types}

        while experiment_count < self.total_space_size:
            start_exp_count = experiment_count
            start_best_conv = best_conversion
            improved_in_pass = False

            # 3 & 4. 遍历每个 reagent_type 进行寻优
            for rt in order:
                rt_idx = self.rt_to_idx[rt]
                results = []

                for candidate in self.candidates[rt]:
                    test_state = current_state.copy()
                    test_state[rt_idx] = candidate
                    key = tuple(test_state)

                    # 6. 不重复遍历：若未测试过，则增加实验计数
                    if key not in tested_conditions:
                        tested_conditions.add(key)
                        conv = self.get_conversion_by_key(key)
                        experiment_count += 1

                        # 发现超过阈值，立刻成功返回
                        if conv > target_threshold:
                            return experiment_count

                        # 更新全局已知最优条件
                        if conv > best_conversion:
                            best_conversion = conv
                            best_state = test_state.copy()
                    else:
                        # 虽然测过，但仍需要取值用于当前轮次的排名
                        conv = self.get_conversion_by_key(key)

                    results.append((candidate, conv))

                # 根据 Conversion 降序排列
                results.sort(key=lambda x: x[1], reverse=True)

                # 获取当前试剂应该选择的排名（0是最优，1是次优...）
                target_idx = rank_target[rt]

                # 如果超出了候选数量，只能用回最优的
                if target_idx >= len(results):
                    target_idx = 0

                chosen_cand, chosen_conv = results[target_idx]

                # 如果条件发生了变化，说明这一轮探索有动作
                if current_state[rt_idx] != chosen_cand:
                    current_state[rt_idx] = chosen_cand
                    improved_in_pass = True

            # 5. 逃逸机制及状态重置
            if best_conversion > start_best_conv:
                # 机制A：在本轮中发现了全新的全局最优峰！
                # 既然找到了新峰，说明之前的逃逸策略生效了。
                # 此时应把所有变量强制重置回寻找最优解（rank=0）的状态，以便在新峰周围进行正常的OFAT局部寻优。
                for rt in order:
                    rank_target[rt] = 0

            elif (not improved_in_pass) or (experiment_count == start_exp_count):
                # 机制B：陷入局部最优，或者陷入已测条件的死循环。
                # 选择目前已知最优条件的条件
                current_state = best_state.copy()

                # 从第一个reagent_type开始，强制它取“次优”（或次次优...），以跳出局部最优
                escaped = False
                for rt in order:
                    if rank_target[rt] + 1 < len(self.candidates[rt]):
                        rank_target[rt] += 1
                        # 类似里程计机制：当高优先级试剂进位时，后续试剂归零
                        idx = order.index(rt)
                        for sub_rt in order[idx + 1 :]:
                            rank_target[sub_rt] = 0
                        escaped = True
                        break

                if not escaped:
                    # 如果所有试剂的所有组合排名都试过了，说明这套搜索策略已经被彻底耗尽却没达到阈值
                    break

        return experiment_count

    def run_monte_carlo(self, target_threshold: float, n_simulations: int = 1000) -> Dict:
        """Run Monte Carlo simulations for a given target threshold."""
        results = []

        # leave=False 意味着跑完一个阈值后进度条会消失，保持控制台整洁
        for i in tqdm(range(n_simulations), desc=f"Threshold {target_threshold: >2}", leave=False, ncols=80):
            n_exp = self.simulate_ofat_run(target_threshold, random_seed=i)
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
    """Main function to run OFAT expected experiments calculation."""
    dataset_path = "../../datasets/HTE_datasets/suzuki_HTE/suzuki_HTE.csv"
    reagent_types = ["solvent", "ligand", "reactant2", "base", "catalyst"]

    # 阈值 85 到 95 (包含95)
    thresholds = list(range(90, 98))
    n_simulations = 1000

    print("Loading dataset and initializing simulator...")
    simulator = OFATSimulator(dataset_path, reagent_types)

    print(f"Dataset loaded: {len(simulator.dataset)} rows")
    print(f"Reagent types: {reagent_types}")
    for rt in reagent_types:
        print(f"  - {rt}: {len(simulator.candidates[rt])} unique candidates")

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

    output_file = output_dir / "ofat_expected_results.json"
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
