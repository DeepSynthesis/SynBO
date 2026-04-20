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
   restart from the first reagent_type (using second-best if already tested, etc.)
6. Avoid duplicate testing
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Set, Optional
import random


class OFATSimulator:
    """Simulator for One-Factor-At-A-Time (OFAT) optimization method."""
    
    def __init__(self, dataset_path: str, reagent_types: List[str], target_col: str = "Conversion"):
        """Initialize the OFAT simulator."""
        self.dataset = pd.read_csv(dataset_path)
        self.reagent_types = reagent_types
        self.target_col = target_col
        
        self.candidates = {}
        for rt in reagent_types:
            self.candidates[rt] = self.dataset[rt].unique().tolist()
        
        self._build_lookup_table()
        
    def _build_lookup_table(self):
        """Build a lookup table for quick conversion value retrieval."""
        self.lookup = {}
        for _, row in self.dataset.iterrows():
            key = tuple(row[rt] for rt in self.reagent_types)
            self.lookup[key] = row[self.target_col]
    
    def get_conversion(self, condition: Dict[str, str]) -> float:
        """Get the conversion value for a given condition."""
        key = tuple(condition[rt] for rt in self.reagent_types)
        return self.lookup.get(key, 0.0)

    def simulate_ofat_run(self, target_threshold: float, random_seed: Optional[int] = None) -> int:
        """Run a single OFAT simulation to find condition with Conversion > target_threshold."""
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        tested_conditions: Set[Tuple] = set()
        best_condition: Dict[str, str] = {}
        best_conversion = -np.inf
        candidate_rankings: Dict[str, List[Tuple[str, float]]] = {}
        current_candidate_idx: Dict[str, int] = {rt: 0 for rt in self.reagent_types}
        
        experiment_count = 0
        max_iterations = 50000
        
        current_condition = {rt: random.choice(self.candidates[rt]) for rt in self.reagent_types}
        
        first_pass_order = self.reagent_types.copy()
        random.shuffle(first_pass_order)
        
        for rt_to_explore in first_pass_order:
            if experiment_count >= max_iterations:
                return experiment_count
            
            results = []
            for candidate in self.candidates[rt_to_explore]:
                test_condition = current_condition.copy()
                test_condition[rt_to_explore] = candidate
                key = tuple(test_condition[r] for r in self.reagent_types)
                if key in tested_conditions:
                    continue
                
                tested_conditions.add(key)
                conv = self.get_conversion(test_condition)
                experiment_count += 1
                results.append((candidate, conv))
                
                if conv > best_conversion:
                    best_conversion = conv
                    best_condition = test_condition.copy()
                if conv > target_threshold:
                    return experiment_count
            
            results.sort(key=lambda x: x[1], reverse=True)
            candidate_rankings[rt_to_explore] = results
            if results:
                current_condition[rt_to_explore] = results[0][0]
                current_candidate_idx[rt_to_explore] = 0
        
        while experiment_count < max_iterations:
            improved = False
            for rt in self.reagent_types:
                if experiment_count >= max_iterations:
                    return experiment_count
                
                rankings = candidate_rankings.get(rt, [])
                current_idx = current_candidate_idx.get(rt, 0)
                
                for next_idx in range(current_idx + 1, len(rankings)):
                    candidate, _ = rankings[next_idx]
                    test_condition = current_condition.copy()
                    test_condition[rt] = candidate
                    key = tuple(test_condition[r] for r in self.reagent_types)
                    if key in tested_conditions:
                        continue
                    
                    tested_conditions.add(key)
                    conv = self.get_conversion(test_condition)
                    experiment_count += 1
                    
                    if conv > self.get_conversion(current_condition):
                        current_condition[rt] = candidate
                        current_candidate_idx[rt] = next_idx
                        improved = True
                        
                        if conv > best_conversion:
                            best_conversion = conv
                            best_condition = test_condition.copy()
                        if conv > target_threshold:
                            return experiment_count
                        break
            
            if not improved:
                break
        
        return experiment_count

    def run_monte_carlo(self, target_threshold: float, n_simulations: int = 1000) -> Dict:
        """Run Monte Carlo simulations for a given target threshold."""
        results = []
        for i in range(n_simulations):
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
            "percentile_25": float(np.percentile(results, 25)),
            "percentile_75": float(np.percentile(results, 75)),
            "percentile_90": float(np.percentile(results, 90)),
            "percentile_95": float(np.percentile(results, 95)),
            "all_results": results.tolist()
        }


def main():
    """Main function to run OFAT expected experiments calculation."""
    dataset_path = "/home/tzz/.cline/worktrees/bdca9/synbo/benchmark/datasets/HTE_datasets/suzuki_HTE/suzuki_HTE.csv"
    reagent_types = ["solvent", "ligand", "reactant2", "base", "catalyst"]
    thresholds = list(range(85, 96))
    n_simulations = 100
    
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
        print(f"  25th percentile: {result['percentile_25']:.2f}")
        print(f"  75th percentile: {result['percentile_75']:.2f}")
        print(f"  90th percentile: {result['percentile_90']:.2f}")
        print(f"  95th percentile: {result['percentile_95']:.2f}")
    
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
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
