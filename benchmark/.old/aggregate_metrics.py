#!/usr/bin/env python3
"""
Aggregate all benchmark results into a single CSV file.
"""

import json
import os
import csv
from pathlib import Path


def extract_parameters(config):
    """Extract relevant parameters from config.json"""
    params = {}

    # Basic experiment info
    params["experiment_name"] = config.get("experiment_name", "")
    params["base_seed"] = config.get("base_seed", "")
    params["num_rounds"] = config.get("num_rounds", "")
    params["iterations"] = config.get("iterations", "")
    params["batch_size"] = config.get("batch_size", "")

    # Dataset info (extract from path)
    dataset_file = config.get("data_paths", {}).get("dataset_file", "")
    if dataset_file:
        dataset_name = dataset_file.split("/")[-2]  # e.g., "B-H_HTE"
        params["dataset"] = dataset_name
    else:
        params["dataset"] = ""

    # Optimization settings
    opt_settings = config.get("optimization_settings", {})
    params["opt_type"] = opt_settings.get("opt_type", "")
    params["sampling_method"] = opt_settings.get("sampling_method", "")
    params["refine_desc"] = opt_settings.get("refine_desc", "")
    params["optimize_method"] = opt_settings.get("optimize_method", "")

    # Model and acquisition function
    kwargs = opt_settings.get("kwargs", {})
    params["surrogate_model"] = kwargs.get("surrogate_model", "")
    params["acq_func"] = kwargs.get("acq_func", "")

    # Optimization metrics and directions
    opt_metrics = opt_settings.get("opt_metrics", [])
    params["opt_metrics"] = ", ".join(opt_metrics)

    opt_direct_info = opt_settings.get("opt_direct_info", [])
    directions = []
    for info in opt_direct_info:
        opt_direct = info.get("opt_direct", "")
        opt_range = info.get("opt_range", [])
        metric_weight = info.get("metric_weight", 1.0)
        directions.append(f"{opt_direct}({opt_range}, w={metric_weight})")
    params["opt_directions"] = "; ".join(directions)

    # Descriptor normalization
    params["desc_normalize"] = opt_settings.get("desc_normalize", "")

    return params


def extract_metrics(metrics_summary):
    """Extract metrics from metrics_summary.json"""
    metrics = {}

    # AOT_HV: Average Optimal Targets - Hypervolume
    aot_hv = metrics_summary.get("average_optimal_targets", {}).get("hypervolume", {})
    aot_hv_avg = aot_hv.get("average_normalized", 0)
    aot_hv_std = aot_hv.get("std_normalized", 0)
    metrics["AOT_HV"] = f"{aot_hv_avg:.3f}±{aot_hv_std:.3f}"

    # AUC_HV: Area Under Curve - Hypervolume
    auc_hv = metrics_summary.get("auc_of_optimization", {}).get("hypervolume", {})
    auc_hc_avg = auc_hv.get("average_normalized", 0)
    auc_hc_std = auc_hv.get("std_normalized", 0)
    metrics["AUC_HV"] = f"{auc_hc_avg:.3f}±{auc_hc_std:.3f}"

    # Additional yield and cost metrics for reference
    aot_yield = metrics_summary.get("average_optimal_targets", {}).get("yield", {})
    metrics["AOT_yield"] = f"{aot_yield.get('average_normalized', 0):.3f}±{aot_yield.get('std_normalized', 0):.3f}"

    aot_cost = metrics_summary.get("average_optimal_targets", {}).get("cost", {})
    metrics["AOT_cost"] = f"{aot_cost.get('average_normalized', 0):.3f}±{aot_cost.get('std_normalized', 0):.3f}"

    auc_yield = metrics_summary.get("auc_of_optimization", {}).get("yield", {})
    metrics["AUC_yield"] = f"{auc_yield.get('average_normalized', 0):.3f}±{auc_yield.get('std_normalized', 0):.3f}"

    auc_cost = metrics_summary.get("auc_of_optimization", {}).get("cost", {})
    metrics["AUC_cost"] = f"{auc_cost.get('average_normalized', 0):.3f}±{auc_cost.get('std_normalized', 0):.3f}"

    return metrics


def aggregate_results(results_dir="benchmark/results", output_file="benchmark/benchmark_results_summary.csv"):
    """Aggregate all results from the results directory"""

    # Get all subdirectories
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Error: Results directory '{results_dir}' does not exist.")
        return

    subdirs = [d for d in results_path.iterdir() if d.is_dir()]
    print(f"Found {len(subdirs)} result directories")

    # Collect all data
    all_data = []

    for subdir in sorted(subdirs):
        config_file = subdir / "config.json"
        metrics_file = subdir / "metrics_summary.json"

        # Skip if either file is missing
        if not config_file.exists() or not metrics_file.exists():
            print(f"Skipping {subdir.name}: missing config.json or metrics_summary.json")
            continue

        try:
            # Read config
            with open(config_file, "r") as f:
                config = json.load(f)

            # Read metrics
            with open(metrics_file, "r") as f:
                metrics_summary = json.load(f)

            # Extract parameters and metrics
            params = extract_parameters(config)
            metrics = extract_metrics(metrics_summary)

            # Combine
            row = {**params, **metrics}
            all_data.append(row)

            print(f"Processed {subdir.name}")

        except Exception as e:
            print(f"Error processing {subdir.name}: {e}")
            continue

    if not all_data:
        print("No valid results found.")
        return

    # Write to CSV
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Define column order (parameters first, then metrics)
    fieldnames = [
        "dataset",
        "experiment_name",
        "base_seed",
        "num_rounds",
        "iterations",
        "batch_size",
        "opt_metrics",
        "opt_directions",
        "opt_type",
        "desc_normalize",
        "sampling_method",
        "refine_desc",
        "optimize_method",
        "surrogate_model",
        "acq_func",
        "AOT_HV",
        "AUC_HV",
        "AOT_yield",
        "AOT_cost",
        "AUC_yield",
        "AUC_cost",
    ]

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_data)

    print(f"\nAggregated {len(all_data)} results to {output_file}")


if __name__ == "__main__":
    aggregate_results()
