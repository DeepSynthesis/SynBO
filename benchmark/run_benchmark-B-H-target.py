import json
import faulthandler

faulthandler.enable()
import pandas as pd
import seaborn as sns
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from utils.plots import (
    plot_final_distribution_boxplot,
    plot_hypervolume_coverage,
    plot_optimization_curves,
    plot_optimization_process_scatter,
)
from utils.metrics import get_average_optimal_targets, get_auc_of_opt, get_average_optimal_targets_hv, get_auc_of_opt_hv
from synbo import ReactionOptimizer
from synbo.utils import load_desc_dict, get_prev_rxn

# =================================================CONFIG=================================================
global_dir = Path(__file__).parent
data_dir = global_dir / Path("datasets/HTE_datasets/")

NUM_ROUNDS = 10
RECALC = False

CONFIG = {
    "experiment_name": "B-H_Optimization",
    "base_seed": 199,
    "num_rounds": NUM_ROUNDS,
    "max_iterations": 50,
    "batch_size": 5,
    "data_paths": {
        "dataset_file": str(data_dir / "B-H_HTE/B-H_HTE.csv"),
        "descriptor_dir": str(data_dir / "B-H_HTE/descriptors"),
        "results_base_dir": str(global_dir / "results"),
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
        "hv_target_threshold": 0.95,
        "opt_type": "auto",
        "desc_normalize": "minmax",
        "sampling_method": "random",
        "refine_desc": "pass",
        "optimize_method": "default_BO",
        "device": "cuda:3",
        "kwargs": {"surrogate_model": "GP", "acq_func": "EHVI"},
    },
    "constraint_settings": {
        "enable_constraints": False,
        "constraint_method": "llm",
        "hv_stagnation_rounds": 1,
        "hv_improvement_threshold": 0.01,
        "reduce_ratio": 0.1,
        "llm_model": "gemini-3.1-flash-lite-preview",
        "llm_api_key": "sk-Pnmf5IgIJYMBEY8Z7078E31cAbC8437e83B4DdE3CaA72e78",
        "llm_base_url": "https://aihubmix.com/v1",
        "llm_temperature": 0.0,
    },
}

CONFIG["reaction_space"]["index_col"] = [f"index" for r in CONFIG["reaction_space"]["reagent_types"]]


def fill_done_dir(batch_idx, output_dir, dataset_path):
    candidates = list(output_dir.glob(f"batch-{batch_idx}_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No file found for batch {batch_idx} in {output_dir}")
    file_path = max(candidates, key=lambda p: p.stat().st_mtime)
    current_df = pd.read_csv(file_path)
    opt_metrics = CONFIG["optimization_settings"]["opt_metrics"]
    cols_to_drop = [c for c in opt_metrics if c in current_df.columns]
    current_df.drop(columns=cols_to_drop, inplace=True)
    hte_df = pd.read_csv(dataset_path)
    merged_df = pd.merge(
        current_df,
        hte_df[CONFIG["reaction_space"]["reagent_types"] + CONFIG["optimization_settings"]["opt_metrics"]],
        on=CONFIG["reaction_space"]["reagent_types"],
        how="left",
    )
    merged_df.to_csv(file_path, index=False)
    return file_path


def cleanup_temp_files(run_dir, round_idx):
    for temp_file in run_dir.glob("batch-*.csv"):
        temp_file.unlink()
    json_file = run_dir / "prohibited_reagent.json"
    if json_file.exists():
        json_file.rename(f"prohibited_reagent_{round_idx}.json")


def setup_experiment_dir(base_dir, num_rounds):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "single" if num_rounds == 1 else "multiple"
    dir_name = f"{prefix}_{timestamp}"
    experiment_dir = Path(base_dir) / dir_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def find_existing_experiment(base_dir_str, current_config):
    """
    Find existing experiment directory.
    Returns:
        - (experiment_dir, resume_info): if partial or complete match found
        - (None, None): if no match found

    resume_info = {
        'completed_rounds': list of completed round indices,
        'current_round': current round index (if partial),
        'current_iteration': next iteration to run (if partial),
        'is_complete': bool,
    }
    """
    base_dir = Path(base_dir_str)
    if not base_dir.exists():
        return None, None

    # Remove timestamp and results_base_dir from comparison as they may differ
    config_for_comparison = {k: v for k, v in current_config.items() if k not in ["data_paths"]}
    # For data_paths, only compare the dataset_file and descriptor_dir
    current_cfg_str = json.dumps(config_for_comparison, sort_keys=True)

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

            saved_config_for_comparison = {k: v for k, v in saved_config.items() if k not in ["data_paths"]}
            saved_cfg_str = json.dumps(saved_config_for_comparison, sort_keys=True)

            if current_cfg_str != saved_cfg_str:
                continue

            # Config matches, check completion status
            num_rounds = current_config["num_rounds"]
            completed_rounds = []
            current_round = None
            current_iteration = 0
            is_complete = True

            for r_idx in range(num_rounds):
                expected_file = run_dir / f"all_batches_final_round_{r_idx}.csv"
                if expected_file.exists():
                    completed_rounds.append(r_idx)
                else:
                    is_complete = False
                    if current_round is None:
                        current_round = r_idx

            # If there's a partial round, check how many iterations are done
            if current_round is not None:
                # Find batch files for the current round
                batch_files = list(run_dir.glob("batch-*.csv"))
                if batch_files:
                    # Extract iteration numbers from batch files
                    iterations = []
                    for bf in batch_files:
                        match = bf.name.split("_")[0].replace("batch-", "")
                        try:
                            iterations.append(int(match))
                        except ValueError:
                            continue
                    if iterations:
                        current_iteration = max(iterations) + 1
                        print(f"Found partial progress: Round {current_round}, {current_iteration} iterations completed")
                else:
                    current_iteration = 0

            resume_info = {
                "completed_rounds": completed_rounds,
                "current_round": current_round,
                "current_iteration": current_iteration,
                "is_complete": is_complete,
            }

            if is_complete:
                print(f"Found complete matching experiment: {run_dir.name}")
            else:
                print(f"Found partial matching experiment: {run_dir.name}")
                print(f"  - Completed rounds: {completed_rounds}")
                if current_round is not None:
                    print(f"  - Resuming from: Round {current_round}, Iteration {current_iteration}")

            return run_dir, resume_info

        except Exception as e:
            print(f"Warning: Failed to read config in {run_dir}: {e}")
            continue

    return None, None


def save_checkpoint(experiment_dir, round_idx, iteration, status="running", **extra_data):
    """Save checkpoint to track progress"""
    checkpoint = {"round_idx": round_idx, "iteration": iteration, "status": status, "timestamp": datetime.now().isoformat(), **extra_data}
    checkpoint_path = Path(experiment_dir) / "checkpoint.json"
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=4)


def load_checkpoint(experiment_dir):
    """Load checkpoint if exists"""
    checkpoint_path = Path(experiment_dir) / "checkpoint.json"
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def get_round_batch_files(experiment_dir, round_idx):
    """Get all batch files for a specific round"""
    exp_dir = Path(experiment_dir)
    # For round 0, we need to check if there's a corresponding batch file
    # Batch files are named batch-{iteration}_{date}.csv
    batch_files = sorted(exp_dir.glob("batch-*.csv"))
    return batch_files


def load_start_points(start_point_path):
    with open(start_point_path, "r", encoding="utf-8") as f:
        start_points = json.load(f)
    return start_points


def run_simulation(experiment_dir, desc_dict, condition_dict, resume_info=None):
    """Execute main optimization calculation loop with HV > 0.95 stop condition

    Args:
        experiment_dir: Path to the experiment directory
        desc_dict: Descriptor dictionary
        condition_dict: Condition dictionary
        resume_info: Optional resume information from find_existing_experiment
            {
                'completed_rounds': [...],
                'current_round': int or None,
                'current_iteration': int,
                'is_complete': bool,
            }
    """
    from synbo.utils.hv_calculator import calculate_hypervolume_for_batch

    base_seed = CONFIG["base_seed"]

    with open(experiment_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=4, ensure_ascii=False)

    print(f"Experiment Directory: {experiment_dir}")

    start_point_path = Path(__file__).parent / "datasets/HTE_datasets/B-H_HTE/start_point.json"
    start_points = load_start_points(start_point_path)
    print(f"Loaded start points from: {start_point_path}")

    constraint_settings = CONFIG.get("constraint_settings", {})
    enable_constraints = constraint_settings.get("enable_constraints", False)
    constraint_method = constraint_settings.get("constraint_method", "llm")
    hv_stagnation_rounds = constraint_settings.get("hv_stagnation_rounds", 3)
    hv_improvement_threshold = constraint_settings.get("hv_improvement_threshold", 0.01)
    reduce_ratio = constraint_settings.get("reduce_ratio", 0.3)
    llm_model = constraint_settings.get("llm_model", "gpt-4")
    llm_api_key = constraint_settings.get("llm_api_key", None)
    llm_base_url = constraint_settings.get("llm_base_url", None)
    llm_temperature = constraint_settings.get("llm_temperature", 0.7)

    hv_target_threshold = CONFIG["optimization_settings"].get("hv_target_threshold", 0.95)
    max_iterations = CONFIG.get("max_iterations", 50)

    if enable_constraints:
        print(f"\n[Constraints Enabled]")
        print(f"  - Method: {constraint_method}")
        print(f"  - HV stagnation rounds: {hv_stagnation_rounds}")
        print(f"  - HV improvement threshold: {hv_improvement_threshold:.2%}")

    # Determine starting round
    start_round = 0
    completed_rounds = []
    if resume_info:
        completed_rounds = resume_info.get("completed_rounds", [])
        current_round = resume_info.get("current_round")
        if current_round is not None:
            start_round = current_round
        elif resume_info.get("is_complete"):
            print(f"\n[RESUME] All {CONFIG['num_rounds']} rounds already completed!")
            return

    # Resume partially completed round if needed
    resume_round_iteration = 0
    resume_round_batch_files_map = {}
    resume_hv_history = []
    resume_stagnation_count = 0
    resume_best_hv_so_far = 0.0

    if resume_info and resume_info.get("current_round") is not None:
        resume_round_iteration = resume_info.get("current_iteration", 0)
        if resume_round_iteration > 0:
            # Load existing batch files for the current round
            batch_files = sorted(Path(experiment_dir).glob("batch-*.csv"))
            for bf in batch_files:
                try:
                    iter_num = int(bf.name.split("_")[0].replace("batch-", ""))
                    resume_round_batch_files_map[iter_num] = bf
                except (ValueError, IndexError):
                    continue
            print(f"[RESUME] Loaded {len(resume_round_batch_files_map)} existing batch files for round {start_round}")

            # Try to recover HV history from existing checkpoint
            checkpoint = load_checkpoint(experiment_dir)
            if checkpoint and checkpoint.get("round_idx") == start_round:
                resume_hv_history = checkpoint.get("hv_history", [])
                resume_stagnation_count = checkpoint.get("stagnation_count", 0)
                resume_best_hv_so_far = checkpoint.get("best_hv_so_far", 0.0)
                print(f"[RESUME] Restored checkpoint: HV history length={len(resume_hv_history)}, best_HV={resume_best_hv_so_far:.4f}")

    for round_idx in range(start_round, CONFIG["num_rounds"]):
        # Skip completely finished rounds
        if round_idx in completed_rounds:
            print(f"\n[RESUME] Skipping completed round {round_idx}")
            continue

        current_seed = base_seed + round_idx
        print(f"\n{'='*20} Starting Round {round_idx + 1}/{CONFIG['num_rounds']} (Seed: {current_seed}) {'='*20}")
        print(f"Optimization Goal: Maximize yield until HV >= {hv_target_threshold} (Max Iter: {max_iterations})")

        batch_files_map = {}
        constraints = None
        hv_history = []
        stagnation_count = 0
        best_hv_so_far = 0.0
        threshold_met = False

        # Restore state if resuming this round
        if round_idx == start_round and resume_round_iteration > 0:
            batch_files_map = resume_round_batch_files_map.copy()
            hv_history = resume_hv_history.copy()
            stagnation_count = resume_stagnation_count
            best_hv_so_far = resume_best_hv_so_far
            print(f"[RESUME] Restored state for round {round_idx}: starting from iteration {resume_round_iteration}")
            print(f"[RESUME] Current HV history: {hv_history}")
            print(f"[RESUME] Best HV so far: {best_hv_so_far:.4f}")

        # Calculate starting iteration
        start_iter = resume_round_iteration if round_idx == start_round else 0

        for i in range(start_iter, max_iterations):
            print(f"\n--- Round {round_idx+1} | Iteration {i} ---")

            # Save checkpoint
            save_checkpoint(
                experiment_dir,
                round_idx,
                i,
                status="running",
                hv_history=hv_history,
                stagnation_count=stagnation_count,
                best_hv_so_far=best_hv_so_far,
            )

            sbo = ReactionOptimizer(
                opt_metrics=CONFIG["optimization_settings"]["opt_metrics"],
                opt_metric_settings=CONFIG["optimization_settings"]["opt_direct_info"],
                opt_type=CONFIG["optimization_settings"]["opt_type"],
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
                round_key = f"round{round_idx + 1}"
                start_indices = start_points[round_key]
                print(f"Using start points for {round_key}: indices {start_indices}")
                dataset_df = pd.read_csv(CONFIG["data_paths"]["dataset_file"])
                selected_rows = dataset_df.iloc[start_indices]
                prev_rxn_info = selected_rows[
                    CONFIG["reaction_space"]["reagent_types"] + CONFIG["optimization_settings"]["opt_metrics"]
                ].copy()
                prev_rxn_info["batch"] = -1
                sbo.load_prev_rxn(prev_rxn_info=prev_rxn_info)
                sbo.selected_conditions = prev_rxn_info[sbo.condition_types].values
                sbo.recommend_type = ["explore"] * len(start_indices)
                sbo.pred_mean = None
                sbo.pred_std = None
            else:
                try:
                    hv_result = calculate_hypervolume_for_batch(
                        prev_rxn_info=prev_rxns,
                        opt_metrics=CONFIG["optimization_settings"]["opt_metrics"],
                        opt_metric_settings=CONFIG["optimization_settings"]["opt_direct_info"],
                        batch_id=None,
                        reference_point_multiplier=1.1,
                    )
                    current_hv = hv_result["hv_normalized"]
                except Exception as e:
                    print(f"Warning: Could not calculate HV at iteration {i}: {e}")
                    current_hv = 0.0

                best_hv_so_far = max(best_hv_so_far, current_hv)

                if hv_history:
                    last_hv = hv_history[-1]
                    improvement = ((current_hv - last_hv) / last_hv) * 100 if last_hv > 0 else 0
                    if improvement >= hv_improvement_threshold:
                        stagnation_count = 0
                        print(f"  HV improved by {improvement:.3f}%")
                    else:
                        stagnation_count += 1
                        print(f"  HV stagnated for {stagnation_count} rounds")
                    hv_history.append(current_hv)
                else:
                    hv_history.append(current_hv)
                    stagnation_count = 0
                    print(f"  Initial HV: {current_hv:.4f}")

                should_reduce_space = enable_constraints and stagnation_count >= hv_stagnation_rounds and i > 2
                if should_reduce_space:
                    print(f"\n{'='*10} Generating constraints at iteration {i} {'='*10}")
                    try:
                        constraints = sbo.get_constrains(
                            method=constraint_method,
                            reduce_ratio=reduce_ratio,
                            model=llm_model,
                            api_key=llm_api_key,
                            base_url=llm_base_url,
                            temperature=llm_temperature,
                        )
                        if constraints is not None:
                            total_eliminated = sum(len(vals) for vals in constraints.values())
                            print(f"✓ Constraints generated, eliminated {total_eliminated} reagents")
                            stagnation_count = 0
                        else:
                            constraints = None
                    except Exception as e:
                        print(f"⚠ Error generating constraints: {e}")
                        constraints = None

                use_edbo = CONFIG["optimization_settings"].get("use_edbo", False)
                if use_edbo:
                    sbo.optimize_edbo(
                        batch_size=CONFIG["batch_size"],
                        acquisition_function=CONFIG["optimization_settings"].get("edbo_acquisition", "NoisyEHVI"),
                        init_sampling_method="random",
                    )
                else:
                    sbo.optimize(
                        batch_size=CONFIG["batch_size"],
                        optimize_method=CONFIG["optimization_settings"]["optimize_method"],
                        desc_normalize=CONFIG["optimization_settings"]["desc_normalize"],
                        refine_desc=CONFIG["optimization_settings"]["refine_desc"],
                        device=CONFIG["optimization_settings"]["device"],
                        constraints=constraints,
                        **CONFIG["optimization_settings"]["kwargs"],
                    )

            sbo.save_results()
            saved_path = fill_done_dir(i, experiment_dir, CONFIG["data_paths"]["dataset_file"])
            batch_files_map[i] = saved_path

            # Check if HV threshold is met
            print(f">>> Best HV so far: {best_hv_so_far:.4f} (Target: {hv_target_threshold})")
            if best_hv_so_far >= hv_target_threshold:
                threshold_met = True
                print(f"🎉 SUCCESS! HV threshold reached at Iteration {i}. Stopping current round.")
                break

        if not threshold_met:
            print(f"⚠️ Reached max iterations ({max_iterations}) without meeting HV threshold ({hv_target_threshold}).")

        # Merge data
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

        cleanup_temp_files(experiment_dir, round_idx)
        print(f"Cleaned temp files for round {round_idx}")

        # Mark checkpoint for completed round
        save_checkpoint(experiment_dir, round_idx + 1, 0, status="round_complete")

        # Reset resume iteration for subsequent rounds
        resume_round_iteration = 0


def run_plotting(experiment_dir):
    """Main plotting entry function"""
    experiment_dir = Path(experiment_dir)
    print(f"\n{'='*20} Starting Plotting Phase {'='*20}")
    print(f"Source Directory: {experiment_dir}")
    result_files = sorted(list(experiment_dir.glob("all_batches_final_round_*.csv")))
    if not result_files:
        print("No result files found to plot.")
        return
    all_rounds_dfs = [pd.read_csv(f) for f in result_files]
    direction_tags = [i["opt_direct"] for i in CONFIG["optimization_settings"]["opt_direct_info"]]
    range_tags = [i["opt_range"] for i in CONFIG["optimization_settings"]["opt_direct_info"]]
    valid_targets = CONFIG["optimization_settings"]["opt_metrics"]
    if not valid_targets:
        print(f"Error: None of the target columns found in CSV.")
        return
    sns.set_theme(style="whitegrid")
    plot_optimization_curves(all_rounds_dfs, valid_targets, direction_tags, range_tags, experiment_dir)
    if Path(CONFIG["data_paths"]["dataset_file"]).exists():
        plot_hypervolume_coverage(
            all_rounds_dfs, valid_targets, direction_tags, range_tags, Path(CONFIG["data_paths"]["dataset_file"]), experiment_dir
        )
    else:
        print(f"Skipping HV plot: Full space file not found")
    range_tags_final = [[80, 100], [0.0, 0.1]]
    plot_final_distribution_boxplot(all_rounds_dfs, valid_targets, direction_tags, range_tags_final, experiment_dir)
    plot_optimization_process_scatter(
        all_rounds_dfs, valid_targets, direction_tags, range_tags, Path(CONFIG["data_paths"]["dataset_file"]), experiment_dir
    )
    print("All plotting tasks completed.")


def run_metrics(experiment_dir):
    """Calculate and save metrics for the benchmark results."""
    experiment_dir = Path(experiment_dir)
    print(f"\n{'='*20} Starting Metrics Calculation {'='*20}")
    print(f"Source Directory: {experiment_dir}")

    result_files = sorted(list(experiment_dir.glob("all_batches_final_round_*.csv")))
    if not result_files:
        print("No result files found to calculate metrics.")
        return

    all_rounds_dfs = [pd.read_csv(f) for f in result_files]
    valid_targets = CONFIG["optimization_settings"]["opt_metrics"]
    direction_tags = [i["opt_direct"] for i in CONFIG["optimization_settings"]["opt_direct_info"]]
    range_tags = [i["opt_range"] for i in CONFIG["optimization_settings"]["opt_direct_info"]]

    print("\nCalculating average optimal targets...")
    avg_optimal_results = get_average_optimal_targets(all_rounds_dfs, valid_targets, direction_tags, range_tags)

    print("Calculating AUC of optimization...")
    auc_results = get_auc_of_opt(all_rounds_dfs, valid_targets, direction_tags, range_tags)

    print("Calculating Average Optimal Targets HV...")
    full_space_file = Path(CONFIG["data_paths"]["dataset_file"])
    avg_opt_hv_results = get_average_optimal_targets_hv(all_rounds_dfs, valid_targets, direction_tags, range_tags, full_space_file)

    print("Calculating AUC of Optimization HV...")
    auc_hv_results = get_auc_of_opt_hv(all_rounds_dfs, valid_targets, direction_tags, range_tags, full_space_file)

    metrics_summary = {"average_optimal_targets": {}, "auc_of_optimization": {}}

    for target, data in avg_optimal_results.items():
        metrics_summary["average_optimal_targets"][target] = {
            "average": float(data["average_optimal"]),
            "average_normalized": float(data["average_normalized"]),
            "std": float(data["std"]),
            "std_normalized": float(data["std_normalized"]),
        }

    metrics_summary["average_optimal_targets"]["hypervolume"] = {
        "average_normalized": float(avg_opt_hv_results["average_normalized"]),
        "std_normalized": float(avg_opt_hv_results["std_normalized"]),
        "total_hv": float(avg_opt_hv_results["total_hv"]),
    }

    for target, data in auc_results.items():
        metrics_summary["auc_of_optimization"][target] = {
            "average": float(data["average"]),
            "average_normalized": float(data["average_normalized"]),
            "std": float(data["std"]),
            "std_normalized": float(data["std_normalized"]),
        }

    metrics_summary["auc_of_optimization"]["hypervolume"] = {
        "average": float(auc_hv_results["average"]),
        "average_normalized": float(auc_hv_results["average_normalized"]),
        "std": float(auc_hv_results["std"]),
        "std_normalized": float(auc_hv_results["std_normalized"]),
        "total_hv": float(auc_hv_results["total_hv"]),
    }

    metrics_file = experiment_dir / "metrics_summary.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=4, ensure_ascii=False)
    print(f"Metrics summary saved: {metrics_file}")


def main():
    results_base = CONFIG["data_paths"]["results_base_dir"]

    existing_dir = None
    resume_info = None
    if not RECALC:
        existing_dir, resume_info = find_existing_experiment(results_base, CONFIG)

    # from IPython import embed; embed(); exit()
    if existing_dir and resume_info and resume_info.get("is_complete"):
        print(f"\n[CACHE HIT] Identical experiment found at: {existing_dir}")
        print("Skipping simulation, proceeding directly to metrics...")
        experiment_dir = existing_dir
        should_run_sim = False
    elif existing_dir and resume_info:
        # Partial match - resume from checkpoint
        print(f"\n[RESUME] Partial experiment found at: {existing_dir}")
        print(f"[RESUME] Completed rounds: {resume_info.get('completed_rounds', [])}")
        if resume_info.get("current_round") is not None:
            print(f"[RESUME] Resuming from Round {resume_info['current_round']}, Iteration {resume_info['current_iteration']}")
        experiment_dir = existing_dir
        should_run_sim = True
    else:
        if RECALC:
            print("\n[RECALC] Recalc flag is True. Forcing new simulation.")
        else:
            print("\n[CACHE MISS] No identical experiment found. Starting new simulation.")
        experiment_dir = setup_experiment_dir(results_base, CONFIG["num_rounds"])
        should_run_sim = True
        resume_info = None

    if should_run_sim:
        desc_dict, condition_dict = load_desc_dict(
            reagent_types=CONFIG["reaction_space"]["reagent_types"],
            desc_dir=CONFIG["data_paths"]["descriptor_dir"],
            name_suffix=CONFIG["reaction_space"]["name_suffix"],
            index_col=CONFIG["reaction_space"]["index_col"],
            return_condition_dict=True,
            fillna=True,
        )
        run_simulation(experiment_dir, desc_dict, condition_dict, resume_info=resume_info)

    # run_plotting(experiment_dir)
    run_metrics(experiment_dir)

    print(f"\nTask Complete. Results accessed at: {experiment_dir}")


if __name__ == "__main__":
    main()
