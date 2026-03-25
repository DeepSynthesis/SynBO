import json
from os import unlink
import numpy as np
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
from utils.metrics import get_average_optimal_targets, get_auc_of_opt, get_hypervolume, get_average_optimal_targets_hv, get_auc_of_opt_hv
from synbo import ReactionOptimizer
from synbo.utils import load_desc_dict, get_prev_rxn

# =================================================CONFIG=================================================
global_dir = Path(__file__).parent
data_dir = global_dir / Path("../examples/")

# 控制参数
NUM_ROUNDS = 10  # k值：运行多少轮
RECALC = False  # [New] 如果为 True，强制重新计算；如果为 False，尝试寻找现有结果

CONFIG = {
    "experiment_name": "B-H_Optimization (new?)",
    "base_seed": 199,
    "num_rounds": NUM_ROUNDS,  # [New] 将 NUM_ROUNDS 放入 CONFIG 以便存入 JSON 进行比对
    "iterations": 10,
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
        "opt_type": "auto",
        "desc_normalize": "zscore",
        "sampling_method": "random",
        "refine_desc": "pass",
        "optimize_method": "default_BO",
        "kwargs": {"surrogate_model": "BNN", "acq_func": "EHVI"},
    },
    "constraint_settings": {
        "enable_constraints": False,  # Enable/disable constraint-based space reduction (set True to test constraints)
        "constraint_method": "llm",  # Method for generating constraints (currently only "llm" supported)
        "hv_stagnation_rounds": 1,  # Number of rounds without HV improvement before triggering space reduction (adjustable parameter)
        "hv_improvement_threshold": 0.01,  # Minimum HV improvement percentage to consider as improvement (0.01 = 0.01%)
        "reduce_ratio": 0.1,  # Ratio of reagents to eliminate during space reduction (0.0-1.0, adjustable parameter)
        # Additional LLM parameters
        "llm_model": "gemini-3-flash-preview",  # LLM model to use for analysis (use "gpt-4", "gemini-2.5-flash", etc.)
        "llm_api_key": "sk-Pnmf5IgIJYMBEY8Z7078E31cAbC8437e83B4DdE3CaA72e78",  # OpenAI API key (set to environment variable or provide here)
        "llm_base_url": "https://aihubmix.com/v1",
        "llm_temperature": 0.0,  # Temperature parameter for LLM generation
    },
}

CONFIG["reaction_space"]["index_col"] = [f"index" for r in CONFIG["reaction_space"]["reagent_types"]]
# ===========================================================================================================


def fill_done_dir(batch_idx, output_dir, dataset_path):
    """从数据集中获取 yield 和 cost 并填充到保存的 batch 文件中。"""
    candidates = list(output_dir.glob(f"batch-{batch_idx}_*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No file found for batch {batch_idx} in {output_dir}")

    # 取最新的文件
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


def cleanup_temp_files(run_dir):
    """删除所有的 batch-*.csv 文件和json文件"""
    for temp_file in run_dir.glob("batch-*.csv"):
        temp_file.unlink()

    json_file = run_dir / "prohibited_reagent.json"
    if json_file.exists():
        json_file.unlink()


def setup_experiment_dir(base_dir, num_rounds):
    """根据 k 值设置文件夹名称"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 逻辑：k=1 -> single_{date}, k>1 -> multiple_{date}
    if num_rounds == 1:
        prefix = "single"
    else:
        prefix = "multiple"

    dir_name = f"{prefix}_{timestamp}"
    experiment_dir = Path(base_dir) / dir_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def find_existing_experiment(base_dir_str, current_config):
    """
    [New] 扫描 base_dir 下的所有文件夹，寻找是否存在相同的 config.json
    并且检查数据文件是否完整。
    """
    base_dir = Path(base_dir_str)
    if not base_dir.exists():
        return None

    # 1. 准备用于比较的 Config 字符串 (排序键以确保一致性)
    # 我们只比较 JSON 序列化后的字符串，这样可以忽略内存地址等差异
    current_cfg_str = json.dumps(current_config, sort_keys=True)

    print(f"Scanning {base_dir} for existing experiments...")

    # 2. 遍历 results 目录下的所有子文件夹
    for run_dir in base_dir.iterdir():
        if not run_dir.is_dir():
            continue

        config_path = run_dir / "config.json"
        if not config_path.exists():
            continue

        # 3. 读取已有的 config 并比较
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                saved_config = json.load(f)

            saved_cfg_str = json.dumps(saved_config, sort_keys=True)

            if current_cfg_str == saved_cfg_str:
                # 4. Config 匹配，接下来检查数据完整性
                # 必须包含所有轮次的最终汇总文件
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


def run_simulation(experiment_dir, desc_dict, condition_dict):
    """执行主要的优化计算循环"""
    from synbo.utils.hv_calculator import calculate_hypervolume_for_batch

    base_seed = CONFIG["base_seed"]

    # 保存配置
    with open(experiment_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(CONFIG, f, indent=4, ensure_ascii=False)

    print(f"Experiment Directory: {experiment_dir}")

    # 加载 start_point.json
    start_point_path = Path(__file__).parent / "datasets/HTE_datasets/B-H_HTE/start_point.json"
    start_points = load_start_points(start_point_path)
    print(f"Loaded start points from: {start_point_path}")

    # Initialize constraint settings
    constraint_settings = CONFIG.get("constraint_settings", {})
    enable_constraints = constraint_settings.get("enable_constraints", False)
    constraint_method = constraint_settings.get("constraint_method", "llm")
    hv_stagnation_rounds = constraint_settings.get("hv_stagnation_rounds", 3)
    hv_improvement_threshold = constraint_settings.get("hv_improvement_threshold", 0.01)
    reduce_ratio = constraint_settings.get("reduce_ratio", 0.3)

    # LLM parameters
    llm_model = constraint_settings.get("llm_model", "gpt-4")
    llm_api_key = constraint_settings.get("llm_api_key", None)
    llm_base_url = constraint_settings.get("llm_base_url", None)
    llm_temperature = constraint_settings.get("llm_temperature", 0.7)

    if enable_constraints:
        print(f"\n[Constraints Enabled]")
        print(f"  - Method: {constraint_method}")
        print(f"  - HV stagnation rounds: {hv_stagnation_rounds}")
        print(f"  - HV improvement threshold: {hv_improvement_threshold:.2%}")
        print(f"  - Reduce ratio: {reduce_ratio:.1%}")
        print(f"  - LLM model: {llm_model}")

    for round_idx in range(CONFIG["num_rounds"]):
        current_seed = base_seed + round_idx
        print(f"\n{'='*20} Starting Round {round_idx + 1}/{CONFIG['num_rounds']} (Seed: {current_seed}) {'='*20}")

        batch_files_map = {}  # 存储 batch_id -> file_path，用于最后合并
        constraints = None  # Initialize constraints as None
        hv_history = []  # Track HV values to detect stagnation
        stagnation_count = 0  # Count consecutive rounds without improvement

        for i in tqdm(range(CONFIG["iterations"]), desc=f"Round {round_idx+1} Calc"):
            rxn_opt = ReactionOptimizer(
                opt_metrics=CONFIG["optimization_settings"]["opt_metrics"],
                opt_metric_settings=CONFIG["optimization_settings"]["opt_direct_info"],
                opt_type=CONFIG["optimization_settings"]["opt_type"],
                random_seed=current_seed,
                quiet=False,
                save_dir=str(experiment_dir),
            )

            rxn_opt.load_rxn_space(condition_dict=condition_dict)
            rxn_opt.load_desc(desc_dict=desc_dict)

            if i > 0:
                prev_rxns = get_prev_rxn(file_root_dir=experiment_dir, file_pattern=str("batch-*.csv"))
                rxn_opt.load_prev_rxn(prev_rxn_info=prev_rxns)

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
                    CONFIG["reaction_space"]["reagent_types"] + CONFIG["optimization_settings"]["opt_metrics"]
                ].copy()
                prev_rxn_info["batch"] = -1  # Set batch to -1 for initial batch

                # Load selected reactions as previous reactions
                rxn_opt.load_prev_rxn(prev_rxn_info=prev_rxn_info)

                # Set selected_conditions from prev_rxn_info
                rxn_opt.selected_conditions = prev_rxn_info[rxn_opt.condition_types].values
                rxn_opt.recommend_type = ["explore"] * len(start_indices)
                rxn_opt.pred_mean = None
                rxn_opt.pred_std = None

                print(f"Loaded {len(start_indices)} initial conditions from dataset rows")

            else:
                # Calculate current HV before optimization
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

                # Check HV improvement
                if hv_history:
                    last_hv = hv_history[-1]
                    improvement = ((current_hv - last_hv) / last_hv) * 100 if last_hv > 0 else 0

                    # Check if HV improved beyond threshold
                    if improvement >= hv_improvement_threshold:
                        stagnation_count = 0
                        print(f"  HV improved by {improvement:.3f}% (stagnation count reset)")
                    else:
                        stagnation_count += 1
                        print(f"  HV stagnated for {stagnation_count} rounds (improvement: {improvement:.3f}%)")

                    hv_history.append(current_hv)
                else:
                    # First iteration, just record HV
                    hv_history.append(current_hv)
                    stagnation_count = 0
                    print(f"  Initial HV: {current_hv:.4f}")

                # Generate constraints if enabled and HV has stagnated for k rounds
                should_reduce_space = enable_constraints and stagnation_count >= hv_stagnation_rounds and i > 2

                if should_reduce_space:
                    print(f"\n{'='*10} Generating constraints at iteration {i} (HV stagnated for {stagnation_count} rounds) {'='*10}")

                    try:
                        constraints = rxn_opt.get_constrains(
                            method=constraint_method,
                            reduce_ratio=reduce_ratio,
                            model=llm_model,
                            api_key=llm_api_key,
                            base_url=llm_base_url,
                            temperature=llm_temperature,
                        )

                        if constraints is not None:
                            total_eliminated = sum(len(vals) for vals in constraints.values())
                            print(f"✓ Constraints generated successfully")
                            print(f"  - Eliminated {total_eliminated} reagents total")
                            for ctype, vals in constraints.items():
                                print(f"  - {ctype}: {len(vals)} reagents eliminated")
                            # Reset stagnation count after space reduction
                            stagnation_count = 0
                        else:
                            print(f"⚠ Constraints generation returned None - using full reaction space")
                            constraints = None
                    except Exception as e:
                        print(f"⚠ Error generating constraints: {e}")
                        print(f"  Continuing with full reaction space")
                        constraints = None

                rxn_opt.optimize(
                    batch_size=CONFIG["batch_size"],
                    optimize_method=CONFIG["optimization_settings"]["optimize_method"],
                    desc_normalize=CONFIG["optimization_settings"]["desc_normalize"],
                    refine_desc=CONFIG["optimization_settings"]["refine_desc"],
                    # temperature=CONFIG["optimization_settings"]["temperature"] * (0.9 - i / 10),
                    constraints=constraints,
                    **CONFIG["optimization_settings"]["kwargs"],
                )

            rxn_opt.save_results()

            saved_path = fill_done_dir(i, experiment_dir, CONFIG["data_paths"]["dataset_file"])
            batch_files_map[i] = saved_path

        # --- Round 结束：合并数据 ---
        dfs = []
        # 按顺序读取，并添加 batch_index 列，方便后续绘图（因为临时文件会被删除）
        for b_idx in sorted(batch_files_map.keys()):
            df = pd.read_csv(batch_files_map[b_idx])
            df["batch_index"] = b_idx  # 关键：标记这是第几轮迭代的数据
            df["round_index"] = round_idx  # 关键：标记这是第几次重复实验
            dfs.append(df)

        if dfs:
            df_all = pd.concat(dfs, ignore_index=True)
            final_filename = f"all_batches_final_round_{round_idx}.csv"
            df_all.to_csv(experiment_dir / final_filename, index=False)
            print(f"Saved summary: {final_filename}")

        # --- 清理临时文件 ---
        cleanup_temp_files(experiment_dir)
        print(f"Cleaned temp files for round {round_idx}")


def run_plotting(experiment_dir):
    """
    主绘图入口函数
    """
    experiment_dir = Path(experiment_dir)
    print(f"\n{'='*20} Starting Plotting Phase {'='*20}")
    print(f"Source Directory: {experiment_dir}")
    result_files = sorted(list(experiment_dir.glob("all_batches_final_round_*.csv")))
    if not result_files:
        print("No result files found to plot.")
        return
    # 合并数据
    all_rounds_dfs = [pd.read_csv(f) for f in result_files]
    direction_tags = [i["opt_direct"] for i in CONFIG["optimization_settings"]["opt_direct_info"]]
    range_tags = [i["opt_range"] for i in CONFIG["optimization_settings"]["opt_direct_info"]]

    # 确保列名在 dataframe 中
    valid_targets = CONFIG["optimization_settings"]["opt_metrics"]
    if not valid_targets:
        print(f"Error: None of the target columns {CONFIG['optimization_settings']['opt_metrics']} found in CSV.")
        return
    # 设置 Seaborn 风格
    sns.set_theme(style="whitegrid")
    # --- 1. 绘制优化曲线 (Grid Plot with Variance) ---
    plot_optimization_curves(all_rounds_dfs, valid_targets, direction_tags, range_tags, experiment_dir)
    # --- 2. 绘制超体积占比 (Single Plot) ---
    # 注意：需要提供 CONFIG['data_paths']['dataset_file'] 并且安装 pymoo
    if Path(CONFIG["data_paths"]["dataset_file"]).exists():
        plot_hypervolume_coverage(
            all_rounds_dfs, valid_targets, direction_tags, range_tags, Path(CONFIG["data_paths"]["dataset_file"]), experiment_dir
        )
    else:
        print(f"Skipping HV plot: Full space file not found at {CONFIG['data_paths']['dataset_file']}")
    # --- 3. 绘制最终最佳值分布 (Box Plot) ---
    range_tags_final = [[80, 100], [0.0, 0.1]]
    plot_final_distribution_boxplot(all_rounds_dfs, valid_targets, direction_tags, range_tags_final, experiment_dir)
    plot_optimization_process_scatter(
        all_rounds_dfs, valid_targets, direction_tags, range_tags, Path(CONFIG["data_paths"]["dataset_file"]), experiment_dir
    )
    print("All plotting tasks completed.")


def run_metrics(experiment_dir):
    """
    Calculate and save metrics for the benchmark results.
    """
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

    # 1. 检查是否存在相同的实验记录 [New Logic]
    existing_dir = None
    if not RECALC:
        existing_dir = find_existing_experiment(results_base, CONFIG)

    if existing_dir:
        print(f"\n[CACHE HIT] Identical experiment found at: {existing_dir}")
        print("Skipping simulation, proceeding directly to plotting...")
        experiment_dir = existing_dir
        should_run_sim = False
    else:
        if RECALC:
            print("\n[RECALC] Recalc flag is True. Forcing new simulation.")
        else:
            print("\n[CACHE MISS] No identical experiment found. Starting new simulation.")

        # 创建新的实验目录
        experiment_dir = setup_experiment_dir(results_base, CONFIG["num_rounds"])
        should_run_sim = True

    # 2. 只有在需要计算时才加载描述符 (可以节省IO)
    if should_run_sim:
        desc_dict, condition_dict = load_desc_dict(
            reagent_types=CONFIG["reaction_space"]["reagent_types"],
            desc_dir=CONFIG["data_paths"]["descriptor_dir"],
            name_suffix=CONFIG["reaction_space"]["name_suffix"],
            index_col=CONFIG["reaction_space"]["index_col"],
            return_condition_dict=True,
            fillna=True,
        )
        # 执行计算部分 (Calculation)
        run_simulation(experiment_dir, desc_dict, condition_dict)

    # 3. 执行绘图部分 (Plotting)
    # 无论是复用旧数据还是新计算的数据，都运行绘图，以防旧数据的图被误删或需要更新绘图样式
    # run_plotting(experiment_dir)

    # 4. 执行指标计算部分 (Metrics)
    run_metrics(experiment_dir)

    print(f"\nTask Complete. Results accessed at: {experiment_dir}")


if __name__ == "__main__":
    main()
