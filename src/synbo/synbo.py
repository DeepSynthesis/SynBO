"""Reaction Optimization Framework.

A modern, efficient framework for multi-objective reaction optimization
using Bayesian Optimization techniques.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
import torch

from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule

from synbo.utils.export_data import save_df

from .optimize import Optimizer
from .descriptor.desc_proc import array_process, done_array_process, array_standarization
from .utils.util_func import check_desc_completeness, generate_constraint_mask, generate_onehot_desc, track_called, get_opt_type
from .initialize import Initializer
from .utils.write_excel import ExcelWriter
from .utils.logger import _logger_default, console
from .algorithm.edbo import EDBOplus, EDBOStandardScaler

default_settings = {"opt_direct": "max", "opt_range": [0, 100], "metric_weight": 1.0}


class ReactionOptimizer:
    """Reaction Optimization Framework.

    A sophisticated framework for optimizing chemical reactions using
    Bayesian Optimization .

    Args:
        opt_metrics: Optimization metrics (str or list of str)
        opt_type: Optimization type ("init", "opt", or "auto")

    Raises:
        ValueError: If invalid parameters are provided
    """

    def __init__(
        self,
        opt_metrics: Union[str, List[str]],
        opt_metric_settings: Union[dict, List[dict]] = {"opt_direct": "max", "opt_range": [0, 100], "metric_weight": 1.0},
        opt_type: Literal["init", "opt", "auto"] = "auto",
        random_seed: int = 42,
        quiet: bool = False,
        save_dir: str = None,
    ) -> None:
        if isinstance(opt_metrics, str):
            opt_metrics = [opt_metrics]
        elif not isinstance(opt_metrics, list):
            raise ValueError("opt_metrics must be str or list")

        if isinstance(opt_metric_settings, dict):
            opt_metric_settings = [opt_metric_settings] * len(opt_metrics)
        elif not isinstance(opt_metric_settings, list):
            raise ValueError("opt_metric_settings must be str or list")
        opt_metric_settings = [{**default_settings, **d} for d in opt_metric_settings]

        assert all(type(d) == dict for d in opt_metric_settings), "opt_metric_settings must be dict or list of dict"
        assert all(d["opt_direct"] in ["max", "min"] for d in opt_metric_settings), "opt_direct must be 'max' or 'min'"

        if opt_type not in ["init", "opt", "auto"]:
            raise ValueError("opt_type must be 'init', 'opt' or 'auto'")

        _logger_default._set_quiet(quiet)

        self.condition_dict = {}
        self.desc_dict = {}
        self.opt_metrics = opt_metrics
        self.opt_metric_settings = opt_metric_settings
        self.opt_type = opt_type
        self.batch_id = 0
        self.random_seed = random_seed
        self.quiet = quiet

        # Set global random seeds for reproducibility
        self._set_random_seeds(random_seed)

        self.save_dir = Path(save_dir) if save_dir else Path.cwd()
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.opt_console = console

        self.opt_console.print(
            Panel(
                f"[bold blue]ReactionOptimizer initialized[/bold blue]\n" f"Metrics: {', '.join(self.opt_metrics)}\n" f"Mode: {opt_type}",
                title="🧪 Reaction Optimizer",
                expand=False,
            )
        )

    def _set_random_seeds(self, seed: int) -> None:
        """Set global random seeds for reproducibility.

        Args:
            seed: Random seed to use
        """
        import random

        random.seed(seed)

        np.random.seed(seed)

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def load_rxn_space(self, condition_dict: Dict[str, List[Any]]) -> None:
        """Load reaction condition space.

        Args:
            condition_dict: Dictionary of condition types and their possible values
        """
        # Sort conditions for reproducibility
        try:
            for k, v in condition_dict.items():
                if isinstance(v, pd.Series) or (type(v) == list and type(v[0]) in (str, int, float)):
                    if pd.Series(v).duplicated().sum() > 0:
                        print(pd.Series(v)[pd.Series(v).duplicated()].values)
                    assert pd.Series(v).duplicated().sum() == 0, f"Condition type `{k}` contains duplicate values"
                    condition_dict[k] = sorted(pd.Series(v).fillna("None").drop_duplicates().tolist())
                elif isinstance(v, pd.DataFrame):
                    assert k in v.columns, f"Condition type `{k}` not found in DataFrame!"
                    assert v[k].duplicated().sum() == 0, f"Condition type `{k}` contains duplicate values"
                    condition_dict[k] = sorted(v[k].fillna("None").tolist())
                else:
                    raise TypeError(f"the type of {k} is {type(v)}, which is not supported")
        except Exception as e:
            self.opt_console.print(f"Error: {e}", style="bold red")
            raise Exception(e)

        self.condition_types = list(condition_dict.keys())
        self.condition_dict = condition_dict

        # Display reaction space summary
        table = Table(title="🔬 Reaction Space Summary")
        table.add_column("Condition Type", style="cyan")
        table.add_column("Count", style="magenta", justify="right")
        table.add_column("Sample Values", style="yellow")

        for ctype, values in condition_dict.items():
            sample_str = ", ".join(map(str, values[:3]))
            if len(values) > 3:
                sample_str += "..."
            table.add_row(ctype, str(len(values)), sample_str)

        self.opt_console.print(table)

    def load_desc(self, desc_dict: Optional[Dict[str, Any]] = None) -> None:
        """Load descriptor dictionary.

        Args:
            desc_dict: Optional descriptor dictionary. If None, uses OneHot encoding.

        Raises:
            AssertionError: If condition types don't match
        """
        if desc_dict is None:
            self.opt_console.print(
                "****Warning: No descriptor dictionary provided, using OneHot encoding as alternative!!!****", style="yellow bold"
            )
            self.desc_dict = generate_onehot_desc(self.condition_dict)
        else:
            if set(desc_dict.keys()) != set(self.condition_types):
                raise ValueError("Condition types do not match")
            self.desc_dict = desc_dict

        for k, v in self.desc_dict.items():
            not_numeric_col = [col for col in v.columns if not pd.api.types.is_numeric_dtype(v[col])]
            # 如果v中的某一列不是int或者float之类的数值类型，则删除掉这一列，并且用console打印警告信息
            if not_numeric_col:
                self.opt_console.print(
                    f"🚨 Warning: Non-numeric columns found in descriptors for {k} condition type,"
                    f"including {not_numeric_col}."
                    "Now removing these columns...",
                    style="bold yellow",
                )
            v.drop(columns=not_numeric_col, inplace=True)

        self.opt_console.print("✓ Descriptors loaded successfully", style="green")

    @track_called
    def load_prev_rxn(self, prev_rxn_info: pd.DataFrame, drop_rxn: bool = False) -> None:
        """Load previous reaction information.

        Args:
            prev_rxn_info: DataFrame containing previous reaction data
            drop_rxn: Whether to drop reactions with missing species

        Raises:
            ValueError: If species not found in condition space
        """

        self.opt_type = "opt" if self.opt_type == "auto" else self.opt_type
        self.batch_id = prev_rxn_info["batch"].max() + 1

        # Validate condition types
        missing_types = [t for t in self.condition_types if t not in prev_rxn_info.columns]
        if missing_types:
            raise ValueError(f"Missing condition types: {missing_types}")

        prev_rxn_info[self.condition_types] = prev_rxn_info[self.condition_types].astype(str)
        # Check for missing species in each condition type
        for condition_type in self.condition_types:
            missing_species = set(prev_rxn_info[condition_type]) - set(self.condition_dict[condition_type])

            if missing_species:
                if drop_rxn:
                    self.opt_console.print(
                        f"Warning: {missing_species} not in {condition_type} condition space, dropping these reactions", style="yellow"
                    )
                    prev_rxn_info = prev_rxn_info[~prev_rxn_info[condition_type].isin(missing_species)]
                else:
                    raise ValueError(f"{missing_species} not in {condition_type} condition space")

        # Convert metrics to float
        for opt_metric in self.opt_metrics:
            try:
                prev_rxn_info[opt_metric] = prev_rxn_info[opt_metric].astype(float)
                assert any(np.isnan(prev_rxn_info[opt_metric])) == False
            except:
                raise ValueError("Some of target properties do not have any value or be '[exp_data]'. Check your input previous data.")

        # drop non metric columns
        prev_rxn_info = prev_rxn_info[prev_rxn_info[opt_metric].notna()]

        self.prev_rxn_info = prev_rxn_info
        try:
            assert len(prev_rxn_info) > 0
        except:
            self.opt_console.print("No previous data was loaded. Check input information.", style="red")
            raise ValueError("Cannot input previous data.")
        self.opt_console.print(f"✓ Loaded {len(prev_rxn_info)} previous reactions", style="green")

    def run(
        self, batch_size: int = 5, desc_normalize: Literal["minmax", "zscore", "l2"] = "minmax", expand_rxn_space: bool = False
    ) -> None:
        """Run optimization or initialization.

        Args:
            batch_size: Number of reactions to recommend
            desc_normalize: Descriptor normalization method
            expand_rxn_space: Whether to expand reaction space (future feature)
        """

        if self.opt_type == "auto":
            if getattr(self, "_load_prev_rxn_called", False):
                self.opt_type = "opt"
            else:
                self.opt_type = "init"

        if expand_rxn_space:
            self.opt_console.print("Reaction space expansion not yet implemented", style="yellow bold")

        self.opt_console.print(Rule(title="🚀 Running Calculation", style="bold"))

        self.opt_console.print(
            "Running settings:\n"
            f"_ Optimization type: [bold]{get_opt_type(self.opt_type)}[/bold]\n"
            f"_ Batch size: [bold]{batch_size}[/bold]\n"
            f"_ Normalization: [bold]{desc_normalize}[/bold]\n"
        )

        if self.opt_type == "init":
            self.initialize(batch_size=batch_size, desc_normalize=desc_normalize)
        elif self.opt_type == "opt":
            self.optimize(batch_size=batch_size, desc_normalize=desc_normalize)
        else:
            raise ValueError("opt_type must be 'init' or 'opt'")

    def initialize(
        self,
        batch_size: int = 5,
        desc_normalize: Literal["minmax", "zscore", "l2"] = "minmax",
        sampling_method: Literal["sobol", "random", "lhs", "kmeans"] = "kmeans",
        refine_desc: Literal["auto_select", "filter_only", "pass"] = "auto_select",
    ) -> None:
        """Initialize reaction optimization with initial sampling.

        Args:
            batch_size: Number of initial samples
            desc_normalize: Descriptor normalization method
            sampling_method: Sampling strategy for initial points
        """

        # progress.update(task, description="Checking descriptor completeness...")
        check_desc_completeness(self.desc_dict, self.condition_dict)

        self.total_name_arr, self.total_desc_arr = array_process(
            self.desc_dict, self.condition_dict, self.condition_types, desc_normalize, refine_desc
        )

        initializer = Initializer(numerical_data=self.total_desc_arr, name_data=self.total_name_arr, random_seed=self.random_seed)
        self.selected_conditions = initializer.sampling(method=sampling_method, batch_size=batch_size)

        # All initial points are exploration
        self.recommend_type = ["explore"] * batch_size

        # For initialization, no prediction values available
        self.pred_mean = None
        self.pred_std = None

        self.opt_console.print(
            f"✓ Selected [bold]{batch_size}[/bold] initial conditions using [bold]{sampling_method}[/bold] sampling", style="green"
        )

    def optimize(
        self,
        batch_size: int = 5,
        desc_normalize: Literal["minmax", "zscore", "l2"] = "minmax",
        refine_desc: Literal["auto_select", "filter_only", "pass"] = "auto_select",
        optimize_method: str = "default_BO",
        temperature: float = 0.0,
        constraints: Optional[Dict[str, List[Any]]] = None,
        **optimization_kwargs: Any,
    ) -> None:
        """Optimize reaction conditions using Bayesian Optimization.

        Args:
            batch_size: Number of conditions to recommend
            desc_normalize: Descriptor normalization method
            optimized_method: Optimization algorithm to use
            temperature: Temperature parameter for exploration-exploitation trade-off (0.0 = pure exploitation, higher = more exploration)
            constraints: Dictionary of constraints to apply to the search space. Format: {condition_type: [prohibited_values]}
                        Example: {"base": ["base1", "base2"], "solvent": ["toluene"]}
                        If a condition type is not in the dictionary, all values are allowed.
            opt_weights: Weights for multi-objective optimization
            mc_num_samples: Monte Carlo samples for acquisition function
            max_batch_size: Maximum batch size for acquisition optimization
            gpu_id: GPU device ID to use
        """
        try:
            assert getattr(self, "_load_prev_rxn_called", False) == True
        except:
            self.opt_console.print("Must load previous reaction information before optimization.", style="red")
            raise Exception("No previous reaction information was loaded.")

        # Function 3: Load prohibited reagents from file if they exist
        if self.save_dir:
            from synbo.utils.constraints_io import load_prohibited_reagents, merge_constraints

            file_prohibited = load_prohibited_reagents(self.save_dir)
            if file_prohibited:
                self.opt_console.print("📋 Loading prohibited reagents from file...", style="cyan")
                # Merge with provided constraints
                constraints = merge_constraints(constraints, file_prohibited)
                if constraints:
                    total_prohibited = sum(len(v) for v in constraints.values())
                    self.opt_console.print(f"  Total constraints loaded: {total_prohibited} prohibited reagents", style="cyan")

        check_desc_completeness(self.desc_dict, self.condition_dict)

        self.total_name_arr, self.total_desc_arr = array_process(
            self.desc_dict, self.condition_dict, self.condition_types, "none", refine_desc
        )

        self.done_arr_index = done_array_process(self.prev_rxn_info, self.total_name_arr, self.condition_types)
        self.total_desc_arr = array_standarization(self.total_desc_arr, self.done_arr_index, desc_normalize)

        # TODO: solve this constraints problem!!!!
        done_arr_desc = self.total_desc_arr[self.done_arr_index]
        done_arr_metrics = {k: self.prev_rxn_info[k].values for k in self.opt_metrics}

        # Normalize target values using opt_range
        # self.y_scalers = {}
        # normalized_metrics = {}
        # for i, (metric, opt_metric) in enumerate(zip(self.opt_metrics, self.opt_metric_settings)):
        #     # y_min, y_max = opt_metric["opt_range"]
        #     # self.y_scalers[metric] = {"min": y_min, "max": y_max}

        self.y_scaler = StandardScaler()

        #     # normalized_y = (done_arr_metrics[metric] - y_min) / (y_max - y_min)  # Min-max normalization: (y - y_min) / (y_max - y_min)
        normalized_metrics = self.y_scaler.fit_transform(np.array(list(done_arr_metrics.values())).T).T

        # try:
        import GPUtil

        device_ids = GPUtil.getAvailable(order="memory", limit=1, maxLoad=0.5, maxMemory=0.5, includeNan=False)

        if device_ids:
            device = torch.device(f"cuda:{device_ids[0]}")
        else:
            device = torch.device("cpu")

        # except:

        # device = torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")
        if not torch.cuda.is_available():
            self.opt_console.print("No GPU found. Using CPU instead.", style="yellow")

        optimizer = Optimizer(
            method=optimize_method,
            total_name_data=self.total_name_arr,
            total_desc_arr=self.total_desc_arr,
            random_seed=self.random_seed,
            device=device,
            optimization_kwargs=optimization_kwargs,
        )

        # 替代方案：使用布尔掩码
        mask = np.ones(len(self.total_desc_arr), dtype=bool)
        mask[self.done_arr_index] = False
        candidate_indices = np.where(mask)[0]  # 保持原有顺序

        candidate_X = self.total_desc_arr[candidate_indices]
        candidate_name = self.total_name_arr  # [candidate_indices]

        if constraints is not None and self.total_name_arr is not None and self.condition_types is not None:
            constraint_mask = generate_constraint_mask(
                total_name_arr=self.total_name_arr,
                condition_types=self.condition_types,
                constraints=constraints,
            )

            # progress.log(f"Constraints applied: {constraint_mask.sum()}/{len(constraint_mask)} candidates available", style="cyan")

            # TODO: there is an mask!!
            constraint_mask = constraint_mask[candidate_indices]
        else:
            constraint_mask = None

        self.selected_conditions, self.recommend_type, self.pred_mean, self.pred_std = optimizer.optimize(
            training_X=done_arr_desc,
            training_y=normalized_metrics,
            candidate_X=candidate_X,
            opt_metric_settings=self.opt_metric_settings,
            batch_size=batch_size,
            temperature=temperature,
            constraints=constraint_mask,
            total_name_arr=candidate_name,
            condition_types=self.condition_types,
        )

        # Denormalize prediction values using the same scalers
        if self.pred_mean is not None and self.pred_std is not None:
            # Update the attributes with denormalized values
            self.pred_mean = self.y_scaler.inverse_transform(self.pred_mean)
            self.pred_std = self.pred_std * self.y_scaler.scale_

        # Display optimization summary
        exploit_count = sum(1 for t in self.recommend_type if t == "exploit")
        explore_count = sum(1 for t in self.recommend_type if t == "explore")

        self.opt_console.print(
            Panel(
                f"[green]Optimization Complete![/green]\n"
                f"Recommended: {batch_size} conditions\n"
                f"Exploit: {exploit_count} | Explore: {explore_count}\n"
                f"Method: {optimize_method} | Device: {device}",
                title="🎯 Results Summary",
            )
        )

    def optimize_edbo(
        self,
        batch_size: int = 5,
        acquisition_function: str = "NoisyEHVI",
        init_sampling_method: str = "random",
        init_indices: Optional[List[int]] = None,
        objective_thresholds: Optional[List[float]] = None,
        scaler_features=None,
        scaler_objectives=None,
    ) -> None:
        """Optimize reaction conditions using EDBO+ (Experimental Design via Bayesian Optimization).

        This method implements the EDBO+ algorithm with separate data loading, normalization,
        and optimization steps. It uses Botorch for Bayesian Optimization with EHVI/NoisyEHVI
        acquisition functions for multi-objective optimization.

        Args:
            batch_size: Number of conditions to recommend
            acquisition_function: Acquisition function to use ('EHVI', 'NoisyEHVI', 'EI')
            init_sampling_method: Method for initial sampling when no previous data exists
                ('random', 'with_index', etc.)
            init_indices: Optional list of indices to use for initialization when
                init_sampling_method='with_index'
            objective_thresholds: List of threshold values for each objective.
                If None, uses minimum observed values as reference point.
            scaler_features: Scaler for features (default: MinMaxScaler)
            scaler_objectives: Scaler for objectives (default: EDBOStandardScaler)

        Raises:
            ValueError: If condition space or descriptors are not loaded
        """
        from sklearn.preprocessing import MinMaxScaler

        # Check prerequisites
        if not self.condition_dict:
            raise ValueError("Reaction condition space must be loaded before optimization. Call load_rxn_space() first.")

        if not self.desc_dict:
            raise ValueError("Descriptors must be loaded before optimization. Call load_desc() first.")

        # Determine optimization mode
        has_prev_data = getattr(self, "_load_prev_rxn_called", False) and hasattr(self, "prev_rxn_info")

        if has_prev_data:
            self.opt_type = "opt"
            self.batch_id = self.prev_rxn_info["batch"].max() + 1
        else:
            self.opt_type = "init"

        self.opt_console.print(Rule(title="🚀 Running EDBO+ Optimization", style="bold"))
        self.opt_console.print(
            f"Running settings:\n"
            f"_ Optimization type: [bold]{'Optimization' if has_prev_data else 'Initialization'}[/bold]\n"
            f"_ Batch size: [bold]{batch_size}[/bold]\n"
            f"_ Acquisition function: [bold]{acquisition_function}[/bold]\n"
            f"_ Init sampling: [bold]{init_sampling_method}[/bold]\n"
        )

        # Set default scalers if not provided
        if scaler_features is None:
            scaler_features = MinMaxScaler()
        if scaler_objectives is None:
            scaler_objectives = EDBOStandardScaler()

        # Prepare objective modes from opt_metric_settings
        objective_modes = [setting["opt_direct"] for setting in self.opt_metric_settings]

        # Generate the full reaction scope DataFrame
        # This creates a DataFrame with all possible combinations of conditions
        scope_df = self._generate_scope_dataframe()

        # Create a temporary CSV file for EDBO+
        import tempfile

        temp_dir = tempfile.mkdtemp()
        temp_filename = "edbo_scope.csv"

        # If we have previous data, merge it into the scope
        if has_prev_data:
            scope_df = self._merge_prev_data_into_scope(scope_df)

        # Save scope to temporary CSV
        scope_df.to_csv(f"{temp_dir}/{temp_filename}", index=False)

        # Initialize EDBO+ optimizer
        edbo = EDBOplus()

        # Run EDBO+ optimization
        result_df = edbo.run(
            objectives=self.opt_metrics,
            objective_mode=objective_modes,
            objective_thresholds=objective_thresholds,
            directory=temp_dir,
            filename=temp_filename,
            columns_features="all",  # Use all available features
            batch=batch_size,
            init_sampling_method=init_sampling_method,
            seed=self.random_seed,
            scaler_features=scaler_features,
            scaler_objectives=scaler_objectives,
            acquisition_function=acquisition_function,
            acquisition_function_sampler="SobolQMCNormalSampler",
            init_indices=init_indices,
        )

        # Extract selected conditions from results
        # Priority >= 0.5 indicates selected samples
        selected_df = result_df[result_df["priority"] >= 0.5].copy()

        # Get only the condition type columns (reagent space), not descriptor columns
        # condition_types contains the reagent names like ['base', 'ligand', 'solvent', ...]
        condition_cols = [c for c in self.condition_types if c in selected_df.columns]

        # Also include 'index' column if it exists for tracking
        if "index" in selected_df.columns:
            condition_cols = ["index"] + condition_cols

        self.selected_conditions = scope_df.loc[selected_df.index, ["alkali", "amine", "cobalt", "oxidant", "solvent"]]

        # Store predictions if available
        self.pred_mean = None
        self.pred_std = None

        if has_prev_data and any(f"{m}_predicted_mean" in result_df.columns for m in self.opt_metrics):
            # Extract predictions for selected conditions
            pred_means = []
            pred_stds = []
            for metric in self.opt_metrics:
                mean_col = f"{metric}_predicted_mean"
                std_col = f"{metric}_predicted_std_dev"
                if mean_col in selected_df.columns and std_col in selected_df.columns:
                    pred_means.append(selected_df[mean_col].values)
                    pred_stds.append(selected_df[std_col].values)

            if pred_means:
                self.pred_mean = np.array(pred_means).T
                self.pred_std = np.array(pred_stds).T

        # Set recommendation types (all EDBO+ recommendations are considered exploitation)
        self.recommend_type = ["exploit"] * len(self.selected_conditions)

        # Clean up temporary files
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)

        # Display results
        self.opt_console.print(
            Panel(
                f"[green]EDBO+ Optimization Complete![/green]\n"
                f"Recommended: {len(self.selected_conditions)} conditions\n"
                f"Acquisition: {acquisition_function}",
                title="🎯 Results Summary",
            )
        )

    def _generate_scope_dataframe(self) -> pd.DataFrame:
        """Generate a DataFrame with all possible reaction scope combinations.

        Returns:
            DataFrame with all condition combinations
        """
        from itertools import product

        # Get all condition values
        condition_values = [self.condition_dict[ct] for ct in self.condition_types]

        # Generate all combinations
        all_combinations = list(product(*condition_values))

        # Create DataFrame
        scope_df = pd.DataFrame(all_combinations, columns=self.condition_types)

        # Add index column for tracking
        scope_df["index"] = range(len(scope_df))

        # Add descriptor features for each condition type
        for condition_type in self.condition_types:
            desc_df = self.desc_dict[condition_type]
            # Merge descriptors
            scope_df = scope_df.merge(desc_df, left_on=condition_type, right_index=True, how="left")

        return scope_df

    def _merge_prev_data_into_scope(self, scope_df: pd.DataFrame) -> pd.DataFrame:
        """Merge previous reaction data into the scope DataFrame.

        Args:
            scope_df: DataFrame with reaction scope

        Returns:
            DataFrame with previous data merged
        """
        # Create a copy to avoid modifying original
        result_df = scope_df.copy()

        # Add objective columns with 'PENDING' as default
        for metric in self.opt_metrics:
            result_df[metric] = "PENDING"

        # Merge previous data
        for _, row in self.prev_rxn_info.iterrows():
            # Find matching row in scope
            match_mask = pd.Series([True] * len(result_df))
            for condition_type in self.condition_types:
                match_mask &= result_df[condition_type].astype(str) == str(row[condition_type])

            # Update objective values
            if match_mask.any():
                idx = result_df[match_mask].index[0]
                for metric in self.opt_metrics:
                    result_df.at[idx, metric] = str(row[metric])

        return result_df

    def save_results(
        self,
        filetype: Literal["csv", "excel", "json"] = "csv",
        figure_output: List[str] = None,
        figure_path: Optional[Union[str, Path]] = None,
        suffix: Optional[str] = None,
        transpose: Optional[bool] = False,
    ) -> None:
        """Save recommendations to file.

        Args:
            save_task: Directory path to save results
            filetype: Output file format
            figure_output: List of figure types to generate
            figure_path: Path for figures
            suffix: Optional filename suffix
        """
        # Prepare prediction info dictionary
        pred_info = None
        if self.pred_mean is not None and self.pred_std is not None:
            pred_info = {}
            for i, metric in enumerate(self.opt_metrics):
                pred_info[f"pred {metric}"] = [f"{mean:.2f}±{sigma:.2f}" for mean, sigma in zip(self.pred_mean[:, i], self.pred_std[:, i])]

        save_df(
            save_path=self.save_dir,
            filetype=filetype,
            selected_conditions=self.selected_conditions,
            condition_dict=self.condition_dict,
            recommend_type=self.recommend_type,
            batch_id=self.batch_id,
            pred_info=pred_info,
            opt_metrics=self.opt_metrics,
            figure_output=figure_output,
            figure_path=figure_path,
            suffix=suffix,
            transpose=transpose,
        )

    def get_rxn_space_size(self) -> int:
        """Get the size of the reaction space.

        Returns:
            Size of the reaction space
        """
        size = 1
        for values in self.condition_dict.values():
            size *= len(values)
        return size

    def get_descriptor_shape(self) -> int:
        """Get the shape of the descriptor array.

        Returns:
            Shape of the descriptor array
        """
        if not self.desc_dict:
            raise ValueError("Descriptor dictionary is empty. Load descriptors first.")
        total_shape = 0
        for desc_df in self.desc_dict.values():
            total_shape += desc_df.shape[1]
        return total_shape

    def get_constrains(self, method: str = "llm", **kwargs) -> Optional[Dict[str, List[Any]]]:
        """Get constraints for optimization based on analysis method.

        Args:
            method: Analysis method to use ("llm" or others)
            **kwargs: Additional parameters for the analysis method

        Returns:
            Dictionary of constraints in format {condition_type: [prohibited_values]}
            Returns None if no constraints are needed

        Raises:
            ValueError: If method is not supported or required data is not loaded
        """
        if method == "llm":
            from synbo.analysis.llm_analyzer import LLMAnalyzer
            from synbo.utils.constraints_io import load_prohibited_reagents, save_prohibited_reagents

            if self.prev_rxn_info is None:
                raise ValueError("Previous reaction information must be loaded before getting constraints")

            # Function 2: Load existing prohibited reagents before LLM recommendation
            existing_prohibited = None
            if self.save_dir:
                existing_prohibited = load_prohibited_reagents(self.save_dir)

            analyzer = LLMAnalyzer(
                opt_metrics=self.opt_metrics,
                opt_metric_settings=self.opt_metric_settings,
                prev_rxn=self.prev_rxn_info,
                condition_dict=self.condition_dict,
                existing_prohibited=existing_prohibited,  # Pass existing prohibited to LLM
            )
            constraints = analyzer.analyze(**kwargs)

            # Function 1: Save prohibited reagents after LLM recommendation
            if constraints and self.save_dir:
                save_prohibited_reagents(save_dir=self.save_dir, new_prohibited=constraints, existing_prohibited=existing_prohibited)

            return constraints
        else:
            raise ValueError(f"Unsupported method: {method}. Supported methods: 'llm'")

    def calculate_current_hv(self, batch_id: Optional[int] = None, reference_point_multiplier: float = 1.0) -> Dict[str, any]:
        """Calculate hypervolume (HV) for current optimization progress.

        This method calculates the hypervolume metric for multi-objective optimization,
        which measures the volume of the objective space dominated by the Pareto front.

        Args:
            batch_id: Optional batch ID to calculate HV up to. If None, uses all available data
            reference_point_multiplier: Multiplier for reference point (default: 1.0)

        Returns:
            Dictionary containing:
                - 'hv': Hypervolume value
                - 'hv_normalized': Normalized hypervolume (0 to 1)
                - 'total_hv': Total theoretical hypervolume (reference)
                - 'num_points': Number of points used in calculation
                - 'batch_id': Batch ID used for calculation

        Raises:
            ValueError: If prev_rxn_info has not been loaded
        """
        from synbo.utils.hv_calculator import calculate_hypervolume_for_batch

        if not hasattr(self, "prev_rxn_info") or self.prev_rxn_info is None:
            raise ValueError("Previous reaction information must be loaded before calculating hypervolume. Call load_prev_rxn() first.")

        if batch_id is None:
            # Use the current batch_id if available, otherwise calculate HV for all data
            batch_id = getattr(self, "batch_id", None)

        hv_result = calculate_hypervolume_for_batch(
            prev_rxn_info=self.prev_rxn_info,
            opt_metrics=self.opt_metrics,
            opt_metric_settings=self.opt_metric_settings,
            batch_id=batch_id,
            reference_point_multiplier=reference_point_multiplier,
        )

        return hv_result

    def calculate_hv_by_batch(self, reference_point_multiplier: float = 1.0) -> pd.DataFrame:
        """Calculate hypervolume for each batch cumulatively.

        This method calculates the hypervolume at each batch, including all data
        from previous batches. This shows the progress of optimization over time.

        Args:
            reference_point_multiplier: Multiplier for reference point (default: 1.0)

        Returns:
            DataFrame with columns:
                - 'batch': Batch index
                - 'hv': Hypervolume value
                - 'hv_normalized': Normalized hypervolume (0 to 1)
                - 'num_points': Cumulative number of points

        Raises:
            ValueError: If prev_rxn_info has not been loaded
        """
        from synbo.utils.hv_calculator import calculate_hypervolume_by_batch

        if not hasattr(self, "prev_rxn_info") or self.prev_rxn_info is None:
            raise ValueError("Previous reaction information must be loaded before calculating hypervolume. Call load_prev_rxn() first.")

        hv_results = calculate_hypervolume_by_batch(
            prev_rxn_info=self.prev_rxn_info,
            opt_metrics=self.opt_metrics,
            opt_metric_settings=self.opt_metric_settings,
            reference_point_multiplier=reference_point_multiplier,
        )

        return hv_results
