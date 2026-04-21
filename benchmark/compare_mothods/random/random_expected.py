"""
Random Search Method Expected Experiments Calculation

This script simulates the Random Search method for optimization on different datasets based on a configuration dictionary.
Supports both single-objective (e.g., Conversion) and multi-objective (e.g., Hypervolume).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Set, Optional, Any
import random
from tqdm import tqdm

from synbo.utils.hv_calculator import calculate_hypervolume_by_batch


class RandomSearchSimulator:
    """Simulator for Random Search optimization method."""

    def __init__(
        self,
        dataset_path: str,
        reagent_types: List[str],
        target_col: Optional[str] = None,
        opt_metrics: Optional[List[str]] = None,
        opt_metric_settings: Optional[List[Dict]] = None,
        index_col: str = "index",
    ):
        """Initialize the Random Search simulator."""
        self.dataset = pd.read_csv(dataset_path)
        self.reagent_types = reagent_types
        self.target_col = target_col
        self.opt_metrics = opt_metrics
        self.opt_metric_settings = opt_metric_settings
        self.index_col = index_col
        self.is_multi_objective = opt_metrics is not None and len(opt_metrics) > 1

        # 错误标志位，防止满屏报错
        self._hv_warned = False

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
        except Exception as e:
            if not self._hv_warned:
                print(f"\n[CRITICAL WARNING] HV calculation failed! Error: {e}")
                print("Are you missing 'opt_metric_settings' in your config? Returning 0.0 for now.")
                self._hv_warned = True
            return 0.0

    def simulate_random_search_run(
        self,
        target_threshold: Optional[float] = None,
        hv_threshold: Optional[float] = None,
        random_seed: Optional[int] = None,
    ) -> int:
        """Run a single Random Search simulation (Sampling without replacement)."""
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        # 获取所有可能的实验组合 (基于数据集)
        all_possible_keys = list(self.lookup.keys())

        # 随机打乱搜索空间（这等同于无放回的随机均匀采样）
        random.shuffle(all_possible_keys)

        tested_keys_list: List[Tuple] = []

        for experiment_count, key in enumerate(all_possible_keys, start=1):
            tested_keys_list.append(key)

            if self.is_multi_objective:
                current_global_hv = self.calculate_hypervolume_for_tested(tested_keys_list)
                if hv_threshold is not None and current_global_hv >= hv_threshold:
                    return experiment_count
            else:
                conv = self.get_value_by_key(key)
                if target_threshold is not None and conv >= target_threshold:
                    return experiment_count

        # 如果穷尽了整个搜索空间都没有达到阈值，返回总实验数
        return self.total_space_size

    def run_monte_carlo(
        self,
        target_threshold: Optional[float] = None,
        hv_threshold: Optional[float] = None,
        n_simulations: int = 1000,
    ) -> Dict:
        """Run Monte Carlo simulations."""
        results = []
        desc = f"Threshold {target_threshold}" if target_threshold else f"HV Threshold {hv_threshold:.3f}"

        for i in tqdm(range(n_simulations), desc=desc, leave=False, ncols=80):
            n_exp = self.simulate_random_search_run(
                target_threshold=target_threshold,
                hv_threshold=hv_threshold,
                random_seed=i,
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


def run_benchmark(dataset_name: str, config: Dict[str, Any]) -> Tuple[Dict, List, bool]:
    """Run Random Search benchmark based on dictionary configuration."""

    print("=" * 70)
    print(f"Random Search Benchmark for Dataset: {dataset_name}")
    print("=" * 70)

    is_multi_obj = "opt_metrics" in config
    n_simulations = config.get("n_simulations", 1000)
    thresholds = config.get("thresholds", [])

    print("\nLoading dataset and initializing simulator...")
    simulator = RandomSearchSimulator(
        dataset_path=config["dataset_path"],
        reagent_types=config["reagent_types"],
        target_col=config.get("target_col"),
        opt_metrics=config.get("opt_metrics"),
        opt_metric_settings=config.get("opt_metric_settings"),
        index_col=config.get("index_col", "index"),
    )

    print(f"Dataset loaded: {len(simulator.dataset)} rows")
    print(f"Mode: {'Multi-Objective' if is_multi_obj else 'Single-Objective'}")

    if is_multi_obj:
        print("\nCalculating Global Maximum Hypervolume for this dataset...")
        all_keys = list(simulator.lookup.keys())
        global_max_hv = simulator.calculate_hypervolume_for_tested(all_keys)
        print(f"--> Global Maximum HV (If all experiments are done): {global_max_hv:.4f}")

        for t in thresholds:
            if t > global_max_hv:
                print(f"[Warning] Threshold {t} is GREATER than the global maximum! Random Search will exhaust the search space.")

    all_results = {}
    for threshold in thresholds:
        metric_name = "Hypervolume >=" if is_multi_obj else f"{config.get('target_col', 'Target')} >="
        format_str = f"{threshold:.3f}" if is_multi_obj else f"{threshold}"

        print(f"\n{'='*60}")
        print(f"Running Monte Carlo simulations for {metric_name} {format_str}")
        print(f"{'='*60}")

        kwargs = {"n_simulations": n_simulations}
        if is_multi_obj:
            kwargs["hv_threshold"] = threshold
        else:
            kwargs["target_threshold"] = threshold

        result = simulator.run_monte_carlo(**kwargs)
        all_results[threshold] = result

        print(f"\nResults for {metric_name} {format_str}:")
        print(f"  Mean experiments needed: {result['mean']:.2f} ± {result['std']:.2f}")
        print(f"  Median experiments needed: {result['median']:.2f}")
        print(f"  Min/Max: {result['min']} / {result['max']}")

    return all_results, thresholds, is_multi_obj


DEFAULT_CONFIG = {
    "suzuki": {
        "dataset_path": "../../datasets/HTE_datasets/suzuki_HTE/suzuki_HTE.csv",
        "reagent_types": ["solvent", "ligand", "reactant2", "reactant1", "base"],
        "target_col": "Conversion",
        "thresholds": list(range(85, 98)),
        "n_simulations": 1000,
    },
    "B-H": {
        "dataset_path": "../../datasets/HTE_datasets/B-H_HTE/B-H_HTE.csv",
        "reagent_types": ["concentration", "temperature", "base", "ligand", "solvent"],
        "opt_metrics": ["yield", "cost"],
        # 若需要，请解除这里的注释并匹配你的 synbo 需要的格式:
        "opt_metric_settings": [
            {"opt_direct": "max", "opt_range": [0, 100], "metric_weight": 1.0},
            {"opt_direct": "min", "opt_range": [0, 0.5], "metric_weight": 1.0},
        ],
        "thresholds": [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92],
        "n_simulations": 100,
    },
}


def save_and_print_results(all_results: Dict, thresholds: List, dataset_name: str, is_multi_obj: bool):
    """Save results to JSON file and print summary."""
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"random_expected_results_{dataset_name}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'='*60}")
    print("All results saved to:", output_file)
    print(f"{'='*60}")
    print("\nSummary Table:")
    threshold_label = "HV Threshold" if is_multi_obj else "Threshold"
    print(f"{threshold_label:<15} {'Mean':<10} {'Median':<10} {'Std':<10} {'Min':<8} {'Max':<8}")
    print("-" * 75)
    for threshold in thresholds:
        r = all_results[threshold]
        t_str = f"{threshold:<15.3f}" if isinstance(threshold, float) else f"{threshold:<15}"
        print(f"{t_str} {r['mean']:<10.2f} {r['median']:<10.2f} {r['std']:<10.2f} {r['min']:<8} {r['max']:<8}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Random Search Benchmark for HTE datasets")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to a JSON configuration file. If not provided, built-in defaults are used."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="Specific dataset key to run from the config. Default runs all datasets in config.",
    )
    args = parser.parse_args()

    if args.config:
        print(f"Loading configuration from {args.config}...")
        with open(args.config, "r") as f:
            full_config = json.load(f)
    else:
        print("No external config provided. Using built-in default configurations.")
        full_config = DEFAULT_CONFIG

    datasets_to_run = list(full_config.keys()) if args.dataset == "all" else [args.dataset]
    for ds_name in datasets_to_run:
        if ds_name not in full_config:
            print(f"Error: Dataset '{ds_name}' not found in configuration. Skipping...")
            continue
        ds_config = full_config[ds_name]
        results, thresholds, is_multi = run_benchmark(ds_name, ds_config)
        save_and_print_results(results, thresholds, ds_name, is_multi)

    print("\n" + "=" * 70)
    print("All benchmarks completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
