"""
OFAT (One-Factor-At-A-Time) Method Expected Experiments Calculation

This script simulates the OFAT method for optimization on different datasets:
- suzuki_HTE: Single objective (Conversion)
- B-H_HTE: Multi-objective (yield, cost) using hypervolume as optimization target

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

from synbo.utils.hv_calculator import calculate_hypervolume_by_batch


class OFATSimulator:
    """Simulator for One-Factor-At-A-Time (OFAT) optimization method."""

    def __init__(
        self,
        dataset_path: str,
        reagent_types: List[str],
        target_col: Optional[str] = None,
        opt_metrics: Optional[List[str]] = None,
        opt_metric_settings: Optional[List[Dict]] = None,
        index_col: str = "index",
    ):
        """Initialize the OFAT simulator."""
        self.dataset = pd.read_csv(dataset_path)
        self.reagent_types = reagent_types
        self.target_col = target_col
        self.opt_metrics = opt_metrics
        self.opt_metric_settings = opt_metric_settings
        self.index_col = index_col
        self.is_multi_objective = opt_metrics is not None and len(opt_metrics) > 1

        # Map reagent types to indices for faster list operations
        self.rt_to_idx = {rt: i for i, rt in enumerate(reagent_types)}

        self.candidates = {}
        for rt in reagent_types:
            self.candidates[rt] = self.dataset[rt].unique().tolist()

        self.total_space_size = len(self.dataset)
        self._build_lookup_table()

    def _build_lookup_table(self):
        """Build a lookup table for quick value retrieval."""
        self.lookup = {}
        for _, row in self.dataset.iterrows():
            key = tuple(row[rt] for rt in self.reagent_types)
            if self.is_multi_objective:
                self.lookup[key] = {metric: row[metric] for metric in self.opt_metrics}
                self.lookup[key]["_index"] = row.get(self.index_col, None)
            else:
                self.lookup[key] = row[self.target_col]

    def get_value_by_key(self, key: Tuple):
        """Get the value(s) for a given condition key."""
        return self.lookup.get(key, {} if self.is_multi_objective else 0.0)

    def calculate_hypervolume_for_tested(self, tested_keys: List[Tuple]) -> float:
        """Calculate hypervolume for tested conditions (for multi-objective)."""
        if not self.is_multi_objective:
            raise ValueError("Hypervolume calculation is only for multi-objective datasets")

        data = []
        for key in tested_keys:
            values = self.get_value_by_key(key)
            row = {"batch": 0}
            for metric in self.opt_metrics:
                row[metric] = values.get(metric, 0)
            data.append(row)

        if len(data) == 0:
            return 0.0

        df = pd.DataFrame(data)

        try:
            hv_results = calculate_hypervolume_by_batch(
                prev_rxn_info=df,
                opt_metrics=self.opt_metrics,
                opt_metric_settings=self.opt_metric_settings,
                reference_point_multiplier=1.0,
                cummax=True,
            )
            return float(hv_results["hv_normalized"].iloc[-1])
        except Exception:
            return 0.0

    def simulate_ofat_run(
        self,
        target_threshold: Optional[float] = None,
        hv_threshold: Optional[float] = None,
        random_seed: Optional[int] = None,
        start_indices: Optional[List[int]] = None,
    ) -> int:
        """Run a single OFAT simulation."""
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        tested_conditions: Set[Tuple] = set()
        tested_keys_list: List[Tuple] = []

        current_state = [random.choice(self.candidates[rt]) for rt in self.reagent_types]

        if self.is_multi_objective:
            best_hv = 0.0
        else:
            best_state = current_state.copy()
            best_conversion = -np.inf

        experiment_count = 0

        order = self.reagent_types.copy()
        random.shuffle(order)

        rank_target = {rt: 0 for rt in self.reagent_types}

        # Handle start indices for B-H dataset
        if start_indices is not None and self.is_multi_objective:
            for idx in start_indices:
                row = self.dataset[self.dataset[self.index_col] == idx]
                if len(row) > 0:
                    key = tuple(row.iloc[0][rt] for rt in self.reagent_types)
                    if key not in tested_conditions:
                        tested_conditions.add(key)
                        tested_keys_list.append(key)
                        experiment_count += 1
                    for i, rt in enumerate(self.reagent_types):
                        current_state[i] = row.iloc[0][rt]

            if len(tested_keys_list) > 0:
                best_hv = self.calculate_hypervolume_for_tested(tested_keys_list)
                if hv_threshold is not None and best_hv >= hv_threshold:
                    return experiment_count

        while experiment_count < self.total_space_size:
            start_exp_count = experiment_count
            start_best_val = best_hv if self.is_multi_objective else best_conversion
            improved_in_pass = False

            for rt in order:
                rt_idx = self.rt_to_idx[rt]
                results = []

                for candidate in self.candidates[rt]:
                    test_state = current_state.copy()
                    test_state[rt_idx] = candidate
                    key = tuple(test_state)

                    if key not in tested_conditions:
                        tested_conditions.add(key)
                        tested_keys_list.append(key)

                        if self.is_multi_objective:
                            experiment_count += 1
                            current_hv = self.calculate_hypervolume_for_tested(tested_keys_list)

                            if hv_threshold is not None and current_hv >= hv_threshold:
                                return experiment_count

                            if current_hv > best_hv:
                                best_hv = current_hv

                            results.append((candidate, current_hv))
                        else:
                            conv = self.get_value_by_key(key)
                            experiment_count += 1

                            if target_threshold is not None and conv > target_threshold:
                                return experiment_count

                            if conv > best_conversion:
                                best_conversion = conv
                                best_state = test_state.copy()

                            results.append((candidate, conv))
                    else:
                        if self.is_multi_objective:
                            results.append((candidate, best_hv))
                        else:
                            conv = self.get_value_by_key(key)
                            results.append((candidate, conv))

                results.sort(key=lambda x: x[1], reverse=True)

                target_idx = rank_target[rt]
                if target_idx >= len(results):
                    target_idx = 0

                chosen_cand, chosen_val = results[target_idx]

                if current_state[rt_idx] != chosen_cand:
                    current_state[rt_idx] = chosen_cand
                    improved_in_pass = True

            current_best = best_hv if self.is_multi_objective else best_conversion
            if current_best > start_best_val:
                for rt in order:
                    rank_target[rt] = 0
            elif not improved_in_pass or experiment_count == start_exp_count:
                if not self.is_multi_objective:
                    current_state = best_state.copy()

                escaped = False
                for rt in order:
                    if rank_target[rt] + 1 < len(self.candidates[rt]):
                        rank_target[rt] += 1
                        idx = order.index(rt)
                        for sub_rt in order[idx + 1 :]:
                            rank_target[sub_rt] = 0
                        escaped = True
                        break

                if not escaped:
                    break

        return experiment_count

    def run_monte_carlo(
        self,
        target_threshold: Optional[float] = None,
        hv_threshold: Optional[float] = None,
        n_simulations: int = 1000,
        start_points_config: Optional[Dict] = None,
    ) -> Dict:
        """Run Monte Carlo simulations."""
        results = []

        desc = f"Threshold {target_threshold}" if target_threshold else f"HV Threshold {hv_threshold:.3f}"

        for i in tqdm(range(n_simulations), desc=desc, leave=False, ncols=80):
            start_indices = None
            if start_points_config is not None:
                round_key = f"round{i + 1}"
                if round_key in start_points_config:
                    start_indices = start_points_config[round_key]

            n_exp = self.simulate_ofat_run(
                target_threshold=target_threshold,
                hv_threshold=hv_threshold,
                random_seed=i,
                start_indices=start_indices,
            )
            results.append(n_exp)

        results = np.array(results)
        return {
            "threshold": target_threshold if target_threshold else hv_threshold,
            "n_simulations": n_simulations,
            "mean": float(np.mean(results)),
            "median": float(np.median(results)),
            "std": float(np.std(results)),
            "min": int(np.min(results)),
            "max": int(np.max(results)),
        }


def run_suzuki_benchmark():
    """Run OFAT benchmark for suzuki_HTE dataset (single objective)."""
    dataset_path = "../../datasets/HTE_datasets/suzuki_HTE/suzuki_HTE.csv"
    reagent_types = ["solvent", "ligand", "reactant2", "base", "catalyst"]

    thresholds = list(range(90, 98))
    n_simulations = 1000

    print("=" * 70)
    print("OFAT Benchmark for Suzuki Dataset (Single Objective - Conversion)")
    print("=" * 70)
    print("\nLoading dataset and initializing simulator...")

    simulator = OFATSimulator(
        dataset_path=dataset_path,
        reagent_types=reagent_types,
        target_col="Conversion",
    )

    print(f"Dataset loaded: {len(simulator.dataset)} rows")
    print(f"Reagent types: {reagent_types}")
    for rt in reagent_types:
        print(f"  - {rt}: {len(simulator.candidates[rt])} unique candidates")

    all_results = {}
    for threshold in thresholds:
        print(f"\n{'='*60}")
        print(f"Running Monte Carlo simulations for Conversion > {threshold}")
        print(f"{'='*60}")

        result = simulator.run_monte_carlo(
            target_threshold=threshold,
            n_simulations=n_simulations,
        )
        all_results[threshold] = result

        print(f"\nResults for Conversion > {threshold}:")
        print(f"  Mean experiments needed: {result['mean']:.2f} ± {result['std']:.2f}")
        print(f"  Median experiments needed: {result['median']:.2f}")
        print(f"  Min/Max: {result['min']} / {result['max']}")

    return all_results, thresholds, "suzuki"


def run_bh_benchmark():
    """Run OFAT benchmark for B-H_HTE dataset (multi-objective with hypervolume)."""
    dataset_path = "../../datasets/HTE_datasets/B-H_HTE/B-H_HTE.csv"
    reagent_types = ["base", "ligand", "solvent"]
    index_col = "index"

    opt_metrics = ["yield", "cost"]
    opt_metric_settings = [
        {"opt_direct": "max", "opt_range": [0, 100]},
        {"opt_direct": "min", "opt_range": [0, 1]},
    ]

    start_point_path = Path(dataset_path).parent / "start_point.json"
    with open(start_point_path, "r") as f:
        start_points = json.load(f)

    hv_thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    n_simulations = 10

    print("=" * 70)
    print("OFAT Benchmark for B-H Dataset (Multi-Objective - Hypervolume)")
    print("=" * 70)
    print("\nLoading dataset and initializing simulator...")

    simulator = OFATSimulator(
        dataset_path=dataset_path,
        reagent_types=reagent_types,
        opt_metrics=opt_metrics,
        opt_metric_settings=opt_metric_settings,
        index_col=index_col,
    )

    print(f"Dataset loaded: {len(simulator.dataset)} rows")
    print(f"Reagent types: {reagent_types}")
    for rt in reagent_types:
        print(f"  - {rt}: {len(simulator.candidates[rt])} unique candidates")
    print(f"Optimization metrics: {opt_metrics}")
    print(f"Start points loaded from: {start_point_path}")

    all_results = {}
    for hv_threshold in hv_thresholds:
        print(f"\n{'='*60}")
        print(f"Running Monte Carlo simulations for Hypervolume >= {hv_threshold:.3f}")
        print(f"{'='*60}")

        result = simulator.run_monte_carlo(
            hv_threshold=hv_threshold,
            n_simulations=n_simulations,
            start_points_config=start_points,
        )
        all_results[hv_threshold] = result

        print(f"\nResults for Hypervolume >= {hv_threshold:.3f}:")
        print(f"  Mean experiments needed: {result['mean']:.2f} ± {result['std']:.2f}")
        print(f"  Median experiments needed: {result['median']:.2f}")
        print(f"  Min/Max: {result['min']} / {result['max']}")

    return all_results, hv_thresholds, "B-H"


def save_results(all_results, thresholds, dataset_name):
    """Save results to JSON file."""
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"ofat_expected_results_{dataset_name}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("All results saved to:", output_file)
    print(f"{'='*60}")

    print("\nSummary Table:")
    if dataset_name == "suzuki":
        print(f"{'Threshold':<12} {'Mean':<10} {'Median':<10} {'Std':<10} {'Min':<8} {'Max':<8}")
        print("-" * 70)
        for threshold in thresholds:
            r = all_results[threshold]
            print(f"{threshold:<12} {r['mean']:<10.2f} {r['median']:<10.2f} {r['std']:<10.2f} {r['min']:<8} {r['max']:<8}")
    else:
        print(f"{'HV Threshold':<12} {'Mean':<10} {'Median':<10} {'Std':<10} {'Min':<8} {'Max':<8}")
        print("-" * 70)
        for threshold in thresholds:
            r = all_results[threshold]
            print(f"{threshold:<12.3f} {r['mean']:<10.2f} {r['median']:<10.2f} {r['std']:<10.2f} {r['min']:<8} {r['max']:<8}")


def main():
    """Main function to run OFAT expected experiments calculation."""
    import argparse

    parser = argparse.ArgumentParser(description="OFAT Benchmark for HTE datasets")
    parser.add_argument(
        "--dataset",
        choices=["suzuki", "B-H", "both"],
        default="both",
        help="Dataset to run benchmark on (default: both)",
    )
    args = parser.parse_args()

    if args.dataset in ["suzuki", "both"]:
        suzuki_results, suzuki_thresholds, _ = run_suzuki_benchmark()
        save_results(suzuki_results, suzuki_thresholds, "suzuki")

    if args.dataset in ["B-H", "both"]:
        bh_results, bh_thresholds, _ = run_bh_benchmark()
        save_results(bh_results, bh_thresholds, "B-H")

    print("\n" + "=" * 70)
    print("All benchmarks completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
