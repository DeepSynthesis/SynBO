import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# ============================================================
# Unified color palette for line elements
# ============================================================
LINE_COLORS = [
    "#7153a1",  # Purple (priority for SynBO)
    "#4f75a3",  # Blue
    "#b45475",  # Red
    "#d3991c",  # Dark Blue
]

# ============================================================
# Unified color palette for fill elements (boxplot, stripplot,
# fill_between, etc.)
# ============================================================
FILL_COLORS = [
    "#dfd3ed",  # Light Purple
    "#cbe0f5",  # Light Blue
    "#f5cbd2",  # Light Red
    "#eec978",  # Light Dark Blue
]

# Global matplotlib rcParams for publication-quality plots
plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 14,
        "axes.titlesize": 0,  # Disabled titles
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
    }
)


def plot_optimization_curves(model_data, target_columns, direction_tags, experiment_dir):
    """
    Plot optimization curves for multiple models with confidence intervals.

    Args:
        model_data: Dictionary with model names as keys and list of DataFrames as values
                    Each DataFrame represents one independent run
        target_columns: List of target column names
        direction_tags: Optimization direction list
        range_tags: Theoretical range list
        experiment_dir: Image save directory
    """
    if not (len(target_columns) == len(direction_tags)):
        raise ValueError("target_columns, direction_tags, and range_tags must have same length.")

    all_best_records = []
    all_actual_records = []

    for model_name, dfs in model_data.items():
        print(f"Processing {len(dfs)} runs for model: {model_name}")

        for run_idx, df in enumerate(dfs):
            df = df.sort_values("batch").copy()

            for col, direction in zip(target_columns, direction_tags):
                # Group by batch and find the best value for each batch
                if direction == "max":
                    batch_best = df.groupby("batch")[col].max()
                elif direction == "min":
                    batch_best = df.groupby("batch")[col].min()
                else:
                    raise ValueError(f"Unknown direction '{direction}' for column '{col}'. Use 'max' or 'min'.")

                # Calculate cumulative best across batches (current and all previous batches)
                if direction == "max":
                    cumulative_best = batch_best.cummax()
                elif direction == "min":
                    cumulative_best = batch_best.cummin()

                # Add cumulative best records (best value up to and including current batch)
                # Include run_idx to distinguish between different runs
                for batch_idx, best_value in cumulative_best.items():
                    all_best_records.append(
                        {"batch": batch_idx, "target": col, "value": best_value, "model": model_name, "run_idx": run_idx}
                    )

                # Add batch best records (best value within each batch only)
                for batch_idx, best_value in batch_best.items():
                    all_actual_records.append(
                        {"batch": batch_idx, "target": col, "value": best_value, "model": model_name, "run_idx": run_idx}
                    )

    best_df = pd.DataFrame(all_best_records)
    actual_df = pd.DataFrame(all_actual_records)

    print(f"Total records for plotting - Cumulative best: {len(best_df)}, Batch best: {len(actual_df)}")

    n_cols = len(target_columns)
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 5), constrained_layout=True)

    if n_cols == 1:
        axes = [axes]

    for i, col in enumerate(target_columns):
        ax = axes[i]
        direction = direction_tags[i]

        col_best_data = best_df[best_df["target"] == col]
        col_actual_data = actual_df[actual_df["target"] == col]

        # Lineplot with model hue for cumulative best values
        sns.lineplot(
            data=col_best_data,
            x="batch",
            y="value",
            hue="model",
            palette=LINE_COLORS,
            ax=ax,
            errorbar=("ci", 95),
            linewidth=2.5,
            marker="o",
            markersize=6,
            legend="full",
            zorder=10,
        )

        # Boxplot with model hue for batch best distribution
        sns.boxplot(
            data=col_actual_data,
            x="batch",
            y="value",
            hue="model",
            palette=FILL_COLORS,
            ax=ax,
            width=0.6,
            fliersize=0,
            linewidth=1.0,
            legend=False,
            zorder=5,
        )

        ax.set_xlabel("Batch")
        ax.set_ylabel(f"Best {col}")

        ax.tick_params(axis="both", which="major")
        ax.tick_params(axis="both", which="minor", width=1, length=4)

        ax.grid(True, linestyle="--", alpha=0.6, linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

        ax.legend(loc="best", framealpha=0.9)

    save_path = experiment_dir / "plot_1_optimization_curves.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot 1 saved: {save_path}")


def plot_hypervolume_coverage(model_data, opt_metrics, direction_tags, full_space_file, experiment_dir):
    """
    Plot hypervolume coverage percentage across all models and runs.

    Similar to plot_optimization_curves:
    1. Boxplot: HV distribution for each batch (across multiple rounds)
    2. Lineplot with errorbar: Cumulative best HV (current and all previous batches)

    Key logic:
    1. Normalize all data to [0, 1] using range_tags.
    2. Convert all objectives to minimization (pymoo HV default).
       - min: normalized_value
       - max: 1.0 - normalized_value
    3. Reference point set to [1.1, 1.1, ...] (slightly above worst value 1.0).

    Args:
        model_data: Dictionary with model names as keys and list of DataFrames as values
        target_columns: List of target column names.
        direction_tags: Optimization direction list, e.g., ['max', 'min', 'max'].
        range_tags: Theoretical range list, List[tuple(min, max)], for normalization.
        full_space_file: Full space data file path.
        experiment_dir: Image save directory (Path object or string).
    """
    from synbo.utils.hv_calculator import calculate_hypervolume_by_batch

    if len(opt_metrics) != len(direction_tags):
        raise Exception()

    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    full_df = pd.read_csv(full_space_file) if str(full_space_file).endswith(".csv") else pd.read_excel(full_space_file)

    # Convert direction_tags and range_tags to opt_metric_settings format for synbo
    opt_metric_settings = [{"opt_direct": direction} for direction in direction_tags]

    for i in range(len(opt_metric_settings)):
        opt_metric_settings[i]["opt_range"] = [full_df[opt_metrics[i]].min(), full_df[opt_metrics[i]].max()]

    # Add batch column if not present for full space (use batch=-1 to indicate it's full space)
    if "batch" not in full_df.columns:
        full_df["batch"] = -1

    # Calculate total hypervolume using synbo metho

    all_best_records = []
    all_actual_records = []

    for model_name, dfs in model_data.items():
        for df in dfs:
            df = df.sort_values("batch")

            hv_by_batch_df = calculate_hypervolume_by_batch(
                prev_rxn_info=df,
                opt_metrics=opt_metrics,
                opt_metric_settings=opt_metric_settings,
                reference_point_multiplier=1.1,
                cummax=False,
            )

            cumulative_best_hv = hv_by_batch_df["hv_normalized"].cummax()

            for batch_idx, hv_value in cumulative_best_hv.items():
                all_best_records.append({"batch": batch_idx, "value": hv_value, "model": model_name})

            # Add batch HV records (HV at each batch)
            for batch_idx, hv_value in hv_by_batch_df["hv_normalized"].items():
                all_actual_records.append({"batch": batch_idx, "value": hv_value, "model": model_name})

    best_df = pd.DataFrame(all_best_records)
    actual_df = pd.DataFrame(all_actual_records)

    plt.figure(figsize=(10, 5), constrained_layout=True)

    # Lineplot with model hue for cumulative best HV
    sns.lineplot(
        data=best_df,
        x="batch",
        y="value",
        hue="model",
        palette=LINE_COLORS,
        errorbar=("ci", 95),
        linewidth=2.5,
        marker="o",
        markersize=6,
        legend="full",
        zorder=10,
    )

    # Boxplot with model hue for batch HV distribution
    sns.boxplot(
        data=actual_df,
        x="batch",
        y="value",
        hue="model",
        palette=FILL_COLORS,
        width=0.6,
        fliersize=0,
        linewidth=1.0,
        legend=False,
        zorder=5,
    )

    plt.xlabel("Batch")
    plt.ylabel("Hypervolume")

    plt.tick_params(axis="both", which="major")
    plt.tick_params(axis="both", which="minor", width=1, length=4)

    plt.grid(True, linestyle="--", alpha=0.6, linewidth=0.8)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    ax.legend(loc="lower right", framealpha=0.9)

    save_path = experiment_dir / "plot_2_hypervolume_coverage.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot 2 saved: {save_path}")


def plot_final_distribution_boxplot(model_data, target_columns, direction_tags, experiment_dir):
    """
    Draw enhanced boxplot：distribution of best values found by all models at the end。

    Args:
        model_data: Dictionary with model names as keys and list of DataFrames as values
        target_columns: List of target column names
        direction_tags: Optimization direction list
        range_tags: Theoretical range list
        experiment_dir: Image save directory
    """
    # 1. Extract final best value for each df
    final_best_records = []
    dir_map = dict(zip(target_columns, direction_tags))

    for model_name, dfs in model_data.items():
        for df_idx, df in enumerate(dfs):
            record = {"run_index": df_idx, "model": model_name}
            for col in target_columns:
                direction = dir_map[col]
                if direction == "max":
                    best_val = df[col].max()
                elif direction == "min":
                    best_val = df[col].min()
                else:
                    raise Exception(f"Unknown direction: {direction}")
                record[col] = best_val
            final_best_records.append(record)

    best_df = pd.DataFrame(final_best_records)

    # 2. Transform data format
    melted_df = best_df.melt(id_vars=["run_index", "model"], value_vars=target_columns, var_name="Target", value_name="Best Value")

    # 3. Initialize plotting
    n_cols = len(target_columns)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5), constrained_layout=True)

    if n_cols == 1:
        axes = [axes]

    for i, col in enumerate(target_columns):
        ax = axes[i]
        direction = direction_tags[i]

        col_data = melted_df[melted_df["Target"] == col]

        sns.boxplot(
            data=col_data,
            x="model",
            y="Best Value",
            palette=FILL_COLORS,
            ax=ax,
            width=0.6,
            fliersize=0,
            linewidth=1.0,
            zorder=5,
        )

        sns.stripplot(
            data=col_data,
            x="model",
            y="Best Value",
            palette=LINE_COLORS,
            ax=ax,
            size=6,
            jitter=0.2,
            alpha=0.7,
            edgecolor="white",
            linewidth=1,
            zorder=10,
        )

        ax.tick_params(axis="x", rotation=45)
        ax.set_xlabel("")
        ax.set_ylabel(f"Best {col}")

        ax.tick_params(axis="both", which="major")
        ax.tick_params(axis="both", which="minor", width=1, length=4)

        ax.grid(True, linestyle="--", alpha=0.6, linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)

        ax.legend(loc="best", framealpha=0.9)

    save_path = experiment_dir / "plot_3_best_value_distribution.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot 3 saved: {save_path}")


def plot_optimization_process_scatter(model_data, target_columns, direction_tags, full_space_file, experiment_dir):
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    n_targets = len(target_columns)
    nds = NonDominatedSorting()

    # ==========================================
    # 1. Process full space data (True Pareto Front)
    # ==========================================
    if str(full_space_file).endswith(".csv"):
        full_df = pd.read_csv(full_space_file)
    else:
        full_df = pd.read_excel(full_space_file)
    full_data_raw = full_df[target_columns].values

    full_sorting_data = full_data_raw.copy()
    for i, direction in enumerate(direction_tags):
        if direction == "max":
            full_sorting_data[:, i] *= -1

    full_fronts = nds.do(full_sorting_data)
    true_pf_indices = full_fronts[0]
    true_pf_data = full_data_raw[true_pf_indices]

    sorted_indices = np.argsort(true_pf_data[:, 0])
    true_pf_data_sorted = true_pf_data[sorted_indices]

    # ==========================================
    # 2. Process Pareto front for each round of all models
    # ==========================================
    all_empirical_pfs = []

    for model_name, dfs in model_data.items():
        model_round_pfs = []
        for df in dfs:
            round_data = df[target_columns].values

            exp_sorting_data = round_data.copy()
            for i, direction in enumerate(direction_tags):
                if direction == "max":
                    exp_sorting_data[:, i] *= -1

            exp_fronts = nds.do(exp_sorting_data)
            empirical_pf_indices = exp_fronts[0]
            empirical_pf_data = round_data[empirical_pf_indices]

            sorted_idx = np.argsort(empirical_pf_data[:, 0])
            model_round_pfs.append(empirical_pf_data[sorted_idx])

        if model_round_pfs:
            all_empirical_pfs.append((model_name, model_round_pfs))

    if n_targets >= 2:
        if n_targets > 2:
            print(f"Warning: {n_targets} objectives detected. Plotting only the first 2 dimensions.")
        _plot_2d_scatter_distribution(
            true_pf_data_sorted,
            all_empirical_pfs,
            target_columns[:2],
            direction_tags[:2],
            experiment_dir,
        )


def _compute_attainment_surface(pf_list, x_grid, dir_x, dir_y, penalty_y):
    """
    Calculate step-like Pareto boundary, strictly maintaining monotonicity.
    If some x is not satisfied，give penalty_y (range value) instead of NaN.
    """
    y_grids = []
    for pf in pf_list:
        pf_x = pf[:, 0]
        pf_y = pf[:, 1]
        y_grid = np.zeros_like(x_grid)

        for i, x_val in enumerate(x_grid):
            # Find all points that satisfy/dominate x_val requirement
            if dir_x == "min":
                valid_mask = pf_x <= x_val
            else:
                valid_mask = pf_x >= x_val

            if not np.any(valid_mask):
                y_grid[i] = penalty_y  # Did not meet requirement, apply penalty
            else:
                valid_y = pf_y[valid_mask]
                if dir_y == "min":
                    y_grid[i] = np.min(valid_y)
                else:
                    y_grid[i] = np.max(valid_y)
        y_grids.append(y_grid)

    return np.array(y_grids)


def _plot_2d_scatter_distribution(true_pf_sorted, all_empirical_pfs, targets, directions, save_dir):
    plt.figure(figsize=(9, 5), constrained_layout=True)
    dir_x, dir_y = directions[0], directions[1]

    # ==========================================
    # Core: calculate global boundary, set plot limits and penalty value
    # ==========================================
    all_points = [true_pf_sorted]
    for _, model_pfs in all_empirical_pfs:
        all_points.extend(model_pfs)
    all_points_stacked = np.vstack(all_points)

    global_min_x, global_min_y = all_points_stacked.min(axis=0)
    global_max_x, global_max_y = all_points_stacked.max(axis=0)

    x_margin = (global_max_x - global_min_x) * 0.05
    y_margin = (global_max_y - global_min_y) * 0.05
    if x_margin == 0:
        x_margin = 1.0
    if y_margin == 0:
        y_margin = 1.0

    # Dense x-axis grid for perfect right angles
    x_grid = np.linspace(global_min_x - x_margin, global_max_x + x_margin, 2000)

    # Set penalty value (inferior solutions far exceeding screen)
    if dir_y == "min":
        penalty_y = global_max_y + 100 * y_margin
    else:
        penalty_y = global_min_y - 100 * y_margin

    # ==========================================
    # Draw distribution shadow and median line for each model
    # ==========================================
    for model_idx, (model_name, model_pfs) in enumerate(all_empirical_pfs):
        line_color = LINE_COLORS[model_idx % len(LINE_COLORS)]
        fill_color = FILL_COLORS[model_idx % len(FILL_COLORS)]

        y_grids = _compute_attainment_surface(model_pfs, x_grid, dir_x, dir_y, penalty_y)

        y_median = np.median(y_grids, axis=0)
        y_lower = np.percentile(y_grids, 0, axis=0)  # 0%
        y_upper = np.percentile(y_grids, 100, axis=0)  # 100%

        plt.fill_between(x_grid, y_lower, y_upper, color=fill_color, alpha=0.15, zorder=3)
        plt.plot(x_grid, y_median, color=line_color, linestyle="-", linewidth=2.2, alpha=0.9, label=model_name, zorder=4)

    # ==========================================
    # Draw true front (True Pareto Front)
    # ==========================================
    true_y_grid = _compute_attainment_surface([true_pf_sorted], x_grid, dir_x, dir_y, penalty_y)[0]

    plt.plot(x_grid, true_y_grid, c="#c71515", linestyle="-", linewidth=2.0, alpha=0.8, label="True Pareto Front", zorder=1)
    # Draw real nodes at actual positions
    plt.scatter(true_pf_sorted[:, 0], true_pf_sorted[:, 1], c="#de6666", edgecolors="k", linewidth=0.8, s=65, alpha=1.0, zorder=5)

    # ==========================================
    # Formatting and beautification
    # ==========================================
    # Force truncate Y-axis range, hide penalty value(lines (shooting to infinity) outside screen
    plt.xlim(global_min_x - x_margin, global_max_x + x_margin)
    plt.ylim(global_min_y - y_margin, global_max_y + y_margin)

    plt.xlabel(f"{targets[0]} ({dir_x})")
    plt.ylabel(f"{targets[1]} ({dir_y})")

    plt.tick_params(axis="both", which="major")

    plt.grid(True, linestyle="--", alpha=0.5, linewidth=0.8)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    plt.legend(loc="best", framealpha=0.9)

    save_path = save_dir / "plot_4_pareto_comparison_distribution.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot 4 saved: {save_path}")
