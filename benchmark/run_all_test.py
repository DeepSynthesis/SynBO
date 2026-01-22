import copy

# Import all necessary functions from run_benchmark.py
from run_benchmark import main as run_benchmark_main, CONFIG as ORIGINAL_CONFIG

# =================================================CONFIG=================================================
# Use the original CONFIG as base template
BASE_CONFIG = copy.deepcopy(ORIGINAL_CONFIG)

# 定义不同的优化方法和初始化策略组合
TEST_VARIATIONS = {
    "descriptor": ["DFT", "RDKitDescriptors", "MorganFP", "OneHot"],
    "surrogate_models": ["GP"],  # ["linear", "GP", "RF", "ensemble"],
    "optimize_methods": ["default_BO"],  # ["evolution", "random_select", "particle_swarm", "default_BO"],
    "acq_functions": ["EHVI"],  # , "UCB", "NEI", "ParEGO"],
    "evolution_methods": ["Thompson", "Standard"],
    "sampling_methods": ["random"],  # , "lhs", "kmeans"],
}
# ============================================================================


def generate_test_configs():
    """生成所有测试配置的组合"""
    configs = []

    # 生成所有可能的组合
    for optimize_method in TEST_VARIATIONS["optimize_methods"]:
        for sampling_method in TEST_VARIATIONS["sampling_methods"]:

            # 根据optimize_method决定是否需要surrogate_model和acq_func
            if optimize_method == "random":
                # random方法不需要kwargs
                config = copy.deepcopy(BASE_CONFIG)
                config["optimization_settings"]["optimize_method"] = optimize_method
                config["optimization_settings"]["sampling_method"] = sampling_method
                config["optimization_settings"]["kwargs"] = {}

                config["experiment_name"] = f"B-H_Optimization_random_{sampling_method}"
                configs.append(config)

            elif optimize_method == "default_BO":
                # default_BO需要surrogate_model和acq_func
                for surrogate_model in TEST_VARIATIONS["surrogate_models"]:
                    for acq_func in TEST_VARIATIONS["acq_functions"]:
                        for desc in TEST_VARIATIONS["descriptor"]:
                            config = copy.deepcopy(BASE_CONFIG)
                            config["optimization_settings"]["optimize_method"] = optimize_method
                            config["optimization_settings"]["sampling_method"] = sampling_method
                            config["optimization_settings"]["kwargs"] = {"surrogate_model": surrogate_model, "acq_func": acq_func}

                            config["experiment_name"] = (
                                f"B-H_Optimization_{surrogate_model}_{optimize_method}_{acq_func}_{sampling_method}_with_{desc}"
                            )
                            configs.append(config)

            elif optimize_method == "evolution":
                for surrogate_model in TEST_VARIATIONS["surrogate_models"]:
                    for evolution_method in TEST_VARIATIONS["evolution_methods"]:
                        config = copy.deepcopy(BASE_CONFIG)
                        config["optimization_settings"]["optimize_method"] = optimize_method
                        config["optimization_settings"]["sampling_method"] = sampling_method
                        config["optimization_settings"]["kwargs"] = {"surrogate_model": surrogate_model, "method": evolution_method}

                        config["experiment_name"] = (
                            f"B-H_Optimization_{surrogate_model}_{optimize_method}_{evolution_method}_{sampling_method}"
                        )
                        configs.append(config)
            else:
                # bayesian需要surrogate_model
                for surrogate_model in TEST_VARIATIONS["surrogate_models"]:
                    config = copy.deepcopy(BASE_CONFIG)
                    config["optimization_settings"]["optimize_method"] = optimize_method
                    config["optimization_settings"]["sampling_method"] = sampling_method
                    config["optimization_settings"]["kwargs"] = {"surrogate_model": surrogate_model}

                    config["experiment_name"] = f"B-H_Optimization_{surrogate_model}_{optimize_method}_{sampling_method}"
                    configs.append(config)

    return configs


def run_single_experiment(config, experiment_idx, total_experiments):
    """运行单个实验配置"""
    print(f"\n{'='*50}")
    print(f"Running Experiment {experiment_idx + 1}/{total_experiments}")
    print(f"Config: {config['experiment_name']}")
    print(f"{'='*50}")

    # 临时替换全局CONFIG
    import run_benchmark

    original_config = run_benchmark.CONFIG
    run_benchmark.CONFIG = config

    try:
        # 直接调用原始的main函数
        run_benchmark_main()
        return config["data_paths"]["results_base_dir"]
    finally:
        # 恢复原始CONFIG
        run_benchmark.CONFIG = original_config


def main():
    """主函数：运行所有测试配置"""
    print("Starting comprehensive optimization method testing...")

    # 生成所有测试配置
    test_configs = generate_test_configs()
    print(f"Generated {len(test_configs)} test configurations.")

    # 运行所有实验
    for i, config in enumerate(test_configs):
        try:
            print(f"\n{'='*50}")
            print(f"Running Experiment {i + 1}/{len(test_configs)}")
            print(f"Config: {config['experiment_name']}")
            print(f"{'='*50}")

            run_single_experiment(config, i, len(test_configs))

        except Exception as e:
            print(f"Error running experiment {i+1}: {e}")
            continue

    print(f"\nAll experiments completed.")


if __name__ == "__main__":
    main()
