"""
Random Baseline Benchmark Script for B-H_HTE
This script runs a random sampling baseline with multiple rounds, 
using the same start points as SynBO for fair comparison.

Usage:
    python benchmark/compare_mothods/random/random_baseline_B-H_HTE.py

Reference:
    Based on benchmark workflow from run_benchmark.py
"""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import numpy as np

from utils.metrics import get_average_optimal_targets, get_auc_of_opt, get_average_optimal_targets_hv, get_auc_of_opt_hv
from synbo.utils import load_desc_dict, get_prev_rxn

# =================================================CONFIG=================================================
global_dir = Path(__file__).parent.parent.parent
data_dir = global_dir / Path("datasets/HTE_datasets/")

# Control parameters
NUM_ROUNDS = 10  # Number of rounds to run
NUM_ITERATIONS = 10  # Number of iterations per round
BATCH_SIZE = 5  # Batch size for each iteration
RECALC = True  # If True, force recalculation

CONFIG = {
    "experiment_name": "Random_Baseline_B-H_HTE",
    "base_seed": 199,
    "num_rounds": NUM_ROUNDS,
    "iterations": NUM_ITERATIONS,
    "batch_size": BATCH_SIZE,
    "data_paths": {
        "dataset_file": str(data_dir / "B-H_HTE/B-H_HTE.csv"),
        "descriptor_dir": str(data_dir / "B-H_HTE/descriptors"),
        "results_base_dir": str(global_dir / "results" / "random_baseline"),
    },
    "reaction_space": {
        "reagent_types": ["concentration", "temperature", "base", "ligand", "solvent"],
        "name_suffix": [None, None, "_dft", "_dft", "_dft"],
    },
    "optimization_settings": {
        "opt_metrics": ["yield", "cost"],
        "opt_direct_info": [
            {"opt_direct": "max", "opt_range": [0, 100], "metric_weight": 1.0},
            {"opt_direct": "min", "opt_range": [0, 0.5], "metric_weight": 1.0},
        ],
    },
}

CONFIG["reaction_space"]["index_col"] = [f"index" for r in CONFIG["reaction_space"]["reagent_types"]]
# ===========================================================================================================


def setup_experiment_dir(base_dir, num_rounds):
    """Set folder name based on num_rounds value"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if num_rounds == 1:
        prefix = "single"
    else:
        prefix = "multiple"

    dir_name = f"random_{prefix}_{timestamp}"
    experiment_dir = Path(base_dir) / dir_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def find_existing_experiment(base_dir_str, current_config):
    """Scan folders to find existing config.json with complete data files."""
    base_dir = Path(base_dir_str)
    if not base_dir.exists():
        return None

    current_cfg_str = json.dumps(current_config, sort_keys=True)
    print(f"Scanning {base_dir} for existing experiments...")

    for run_dir in base_dir.iterdir():
        if not run_dir.is_dir():
            continue

        config_path = run_dir / "config.json"
        if not config_path.exists():
            continue

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                saved_config = json.load(f)

            saved_cfg_str = json.dumps(saved_config, sort_keys=True)

            if current_cfg_str == saved_cfg_str:
                is_complete = True
                num_rounds = current_config["num_rounds"]

                for r_idx in range(num_rounds):
                    expected_file = run_dir / f"all_batches_final_round_{r_idx}.csv"
                    if not expected_file.exists():
                        is_complete = False
                        break

                if is_complete:
                    print(f"Found matching existing experiment: {run_dir.name}")
                    return run_dir

        except Exception as e:
            print(f"Warning: Failed to read config in {run_dir}: {e}")
            continue

    return None


def load_start_points(start_point_path):
    """Load start points from JSON file"""
    with open(start_point_path, "r", encoding="utf-8") as f:
        start_points = json.load(f)
    return start_points


def fill_done_dir(batch_idx, output_dir, dataset_path, reagent_types, opt_metrics):
    """Get yield and cost from dataset and fill into saved batch files."""
    candidates = list(output_dir.glob(f"batch-{batch_idx}_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No file found for batch {batch_idx} in {output_dir}")

    file_path = max(candidates, key=lambda p: p.stat().st_mtime)

    current_df = pd.read_csv(file_path)
    cols_to_drop = [c for c in opt_metrics if c in current_df.columns]
    current_df.drop(columns=cols_to_drop, inplace=True)


    hte_df = pd.read_csv(dataset_path)

    merged_df = pd.merge(
        current_df,
        hte_df[reagent_types + opt_metrics],
        on=reagent_types,
        how="left",
    )
    merged_df.to_csv(file_path, index=False)
    return file_path


def cleanup_temp_files(run_dir, round_idx):
    """Delete all batch-*.csv files"""
    for temp_file in run_dir.glob("batch-*.csv"):
        temp_file.unlink()

    json_file = run_dir / "prohibited_reagent.json"
    if json_file.exists():
        json_file.rename(f"prohibited_reagent_{round_idx}.json")


def run_random_sampling(experiment_dir, condition_dict, desc_dict):
    """Execute random sampling optimization loop"""
    from synbo import ReactionOptimizer

    base_seed = CONFIG["base_seed"]

    # Save config
    with open(experiment_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=4, ensure_ascii=False)

    print(f"Experiment Directory: {experiment_dir}")

    # Load start_point.json
    start_point_path = global_dir / "datasets/HTE_datasets/B-H_HTE/start_point.json"
    start_points = load_start_points(start_point_path)
    print(f"Loaded start points from: {start_point_path}")

    reagent_types = CONFIG["reaction_space"]["reagent_types"]
    opt_metrics = CONFIG["optimization_settings"]["opt_metrics"]
    opt_direct_info = CONFIG["optimization_settings"]["opt_direct_info"]
    batch_size = CONFIG["batch_size"]

    for round_idx in range(CONFIG["num_rounds"]):
        current_seed = base_seed + round_idx
        np.random.seed(current_seed)
        print(f"\n{'='*20} Starting Round {round_idx + 1}/{CONFIG['num_rounds']} (Seed: {current_seed}) {'='*20}")

        batch_files_map = {}

        for i in tqdm(range(CONFIG["iterations"]), desc=f"Round {round_idx+1} Random"):
            sbo = ReactionOptimizer(
                opt_metrics=opt_metrics,
                opt_metric_settings=opt_direct_info,
                opt_type="auto",
                random_seed=current_seed,
                quiet=True,
                save_dir=str(experiment_dir),
            )

            sbo.load_rxn_space(condition_dict=condition_dict)
            sbo.load_desc(desc_dict=desc_dict)

            if i > 0:
                prev_rxns = get_prev_rxn(file_root_dir=experiment_dir, file_pattern=str("batch-*.csv"))
                sbo.load_prev_rxn(prev_rxn_info=prev_rxns)

            if i == 0:
                # Use start points from JSON file for initial batch
                round_key = f"round{round_idx + 1}"
                start_indices = start_points[round_key]
                print(f"Using start points for {round_key}: indices {start_indices}")

                # Load dataset and select rows by indices
                dataset_df = pd.read_csv(CONFIG["data_paths"]["dataset_file"])
                selected_rows = dataset_df.iloc[start_indices]

                # Create prev_rxn_info DataFrame with required columns
                prev_rxn_info = selected_rows[
                    reagent_types + opt_metrics
                ].copy()
                prev_rxn_info["batch"] = -1

                # Load selected reactions as previous reactions
                sbo.load_prev_rxn(prev_rxn_info=prev_rxn_info)

                # Set selected_conditions from prev_rxn_info
                sbo.selected_conditions = prev_rxn_info[sbo.condition_types].values
                sbo.recommend_type = ["explore"] * len(start_indices)
                sbo.pred_mean = None
                sbo.pred_std = None

                print(f"Loaded {len(start_indices)} initial conditions from dataset rows")

            else:
                # Random sampling: sample random batch_size conditions
                available_mask = sbo.available_mask
                available_indices = np.where(~available_mask)[0]

                if len(available_indices) >= batch_size:
                    selected_idx = np.random.choice(available_indices, size=batch_size, replace=False)
                else:
                    # If not enough available, sample from all
                    all_indices = np.arange(len(sbo.available_mask))
                    selected_idx = np.random.choice(all_indices, size=batch_size, replace=False)

                # Get selected conditions
                selected_conditions = sbo.condition_matrix[selected_idx]
                sbo.selected_conditions = selected_conditions
                sbo.recommend_type = ["explore"] * batch_size
                sbo.pred_mean = None
                sbo.pred_std = None

                # Mark as explored
                sbo.available_mask[selected_idx] = True

            sbo.save_results()

            saved_path = fill_done_dir(i, experiment_dir, CONFIG["data_paths"]["dataset_file"], reagent_types, opt_metrics)
            batch_files_map[i] = saved_path

        # --- End of Round: Merge data ---
        dfs = []
        for b_idx in sorted(batch_files_map.keys()):
            df = pd.read_csv(batch_files_map[b_idx])
            df["batch_index"] = b_idx
            df["round_index"] = round_idx
            dfs.append(df)

        if dfs:
            df_all = pd.concat(dfs, ignore_index=True)
            final_filename = f"all_batches_final_round_{round_idx}.csv"
            df_all.to_csv(experiment_dir / final_filename, index=False)
            print(f"Saved summary: {final_filename}")

        # --- Clean up temp files ---
        cleanup_temp_files(experiment_dir, round_idx)
        print(f"Cleaned temp files for round {round_idx}")


def run_metrics(experiment_dir):
    """Calculate and save metrics for the benchmark results."""
    experiment_dir = Path(experiment_dir)
    print(f"\n{'='*20} Starting Metrics Calculation {'='*20}")
    print(f"Source Directory: {experiment_dir}")

    result_files = sorted(list(experiment_dir.glob("all_batches_final_round_*.csv")))
    if not result_files:
        print("No result files found to calculate metrics.")
        return

    # Load data
    all_rounds_dfs = [pd.read_csv(f) for f in result_files]
    valid_targets = CONFIG["optimization_settings"]["opt_metrics"]
    direction_tags = [i["opt_direct"] for i in CONFIG["optimization_settings"]["opt_direct_info"]]
    range_tags = [i["opt_range"] for i in CONFIG["optimization_settings"]["opt_direct_info"]]

    # 1. Calculate average optimal targets
    print("\nCalculating average optimal targets...")
    avg_optimal_results = get_average_optimal_targets(all_rounds_dfs, valid_targets, direction_tags, range_tags)

    # 2. Calculate AUC of optimization
    print("Calculating AUC of optimization...")
    auc_results = get_auc_of_opt(all_rounds_dfs, valid_targets, direction_tags, range_tags)

    # 3. Calculate Average Optimal Targets HV
    print("Calculating Average Optimal Targets HV...")
    full_space_file = Path(CONFIG["data_paths"]["dataset_file"])
    avg_opt_hv_results = get_average_optimal_targets_hv(all_rounds_dfs, valid_targets, direction_tags, range_tags, full_space_file)

    # 4. Calculate AUC of Optimization HV
    print("Calculating AUC of Optimization HV...")
    auc_hv_results = get_auc_of_opt_hv(all_rounds_dfs, valid_targets, direction_tags, range_tags, full_space_file)

    # 5. Compile all metrics into a summary
    metrics_summary = {"average_optimal_targets": {}, "auc_of_optimization": {}}

    # Format average optimal results
    for target, data in avg_optimal_results.items():
        metrics_summary["average_optimal_targets"][target] = {
            "average": float(data["average_optimal"]),
            "average_normalized": float(data["average_normalized"]),
            "std": float(data["std"]),
            "std_normalized": float(data["std_normalized"]),
        }

    # Format Hypervolume Optimal results under average_optimal_targets
    metrics_summary["average_optimal_targets"]["hypervolume"] = {
        "average_normalized": float(avg_opt_hv_results["average_normalized"]),
        "std_normalized": float(avg_opt_hv_results["std_normalized"]),
        "total_hv": float(avg_opt_hv_results["total_hv"]),
    }

    # Format AUC results
    for target, data in auc_results.items():
        metrics_summary["auc_of_optimization"][target] = {
            "average": float(data["average"]),
            "average_normalized": float(data["average_normalized"]),
            "std": float(data["std"]),
            "std_normalized": float(data["std_normalized"]),
        }

    # Format Hypervolume AUC results under auc_of_optimization
    metrics_summary["auc_of_optimization"]["hypervolume"] = {
        "average": float(auc_hv_results["average"]),
        "average_normalized": float(auc_hv_results["average_normalized"]),
        "std": float(auc_hv_results["std"]),
        "std_normalized": float(auc_hv_results["std_normalized"]),
        "total_hv": float(auc_hv_results["total_hv"]),
    }

    # 6. Save metrics summary to JSON
    metrics_file = experiment_dir / "metrics_summary.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4, ensure_ascii=False)
    print(f"Metrics summary saved: {metrics_file}")


def main():
    results_base = CONFIG["data_paths"]["results_base_dir"]

    # 1. Check if identical experiment record exists
    existing_dir = None
    if not RECALC:
        existing_dir = find_existing_experiment(results_base, CONFIG)

    if existing_dir:
        print(f"\n[CACHE HIT] Identical experiment found at: {existing_dir}")
        print("Skipping simulation, proceeding directly to metrics...")
        experiment_dir = existing_dir
        should_run_sim = False
    else:
        if RECALC:
            print("\n[RECALC] Recalc flag is True. Forcing new simulation.")
        else:
            print("\n[CACHE MISS] No identical experiment found. Starting new simulation.")

        # Create new experiment directory
        experiment_dir = setup_experiment_dir(results_base, CONFIG["num_rounds"])
        should_run_sim = True

    # 2. Load descriptors only when needed
    if should_run_sim:
        desc_dict, condition_dict = load_desc_dict(
            reagent_types=CONFIG["reaction_space"]["reagent_types"],
            desc_dir=CONFIG["data_paths"]["descriptor_dir"],
            name_suffix=CONFIG["reaction_space"]["name_suffix"],
            index_col=CONFIG["reaction_space"]["index_col"],
            return_condition_dict=True,
            fillna=True,
        )
        # Execute random sampling
        run_random_sampling(experiment_dir, condition_dict, desc_dict)

    # 3. Execute metrics calculation
    run_metrics(experiment_dir)

    print(f"\nTask Complete. Results saved at: {experiment_dir}")


if __name__ == "__main__":
    main()
