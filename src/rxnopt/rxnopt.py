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
import torch

from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule

from rxnopt.utils.export_data import save_df

from .optimize import Optimizer
from .descriptor.desc_proc import array_process, done_array_process
from .utils.util_func import check_desc_completeness, generate_onehot_desc, track_called, get_opt_type
from .initialize import Initializer
from .utils.write_excel import ExcelWriter
from .utils.logger import _logger_default, console

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

        self.condition_dict: Dict[str, List[Any]] = {}
        self.desc_dict: Dict[str, Any] = {}
        self.opt_metrics = opt_metrics
        self.opt_metric_settings = opt_metric_settings
        self.opt_type = opt_type
        self.prev_rxn_info: Optional[pd.DataFrame] = None
        self.batch_id = 0
        self.random_seed = random_seed
        self.quiet = quiet
        self.opt_console = console

        self.opt_console.print(
            Panel(
                f"[bold blue]ReactionOptimizer initialized[/bold blue]\n" f"Metrics: {', '.join(self.opt_metrics)}\n" f"Mode: {opt_type}",
                title="🧪 Reaction Optimizer",
                expand=False,
            )
        )

    def load_rxn_space(self, condition_dict: Dict[str, List[Any]]) -> None:
        """Load reaction condition space.

        Args:
            condition_dict: Dictionary of condition types and their possible values
        """
        # Sort conditions for reproducibility
        try:
            for k, v in condition_dict.items():
                if isinstance(v, pd.Series) or (type(v) == list and type(v[0]) in (str, int, float)):
                    condition_dict[k] = sorted(pd.Series(v).fillna("None").tolist())
                elif isinstance(v, pd.DataFrame):
                    assert k in v.columns, f"Condition type `{k}` not found in DataFrame!"
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
            f"· Optimization type: [bold]{get_opt_type(self.opt_type)}[/bold]\n"
            f"· Batch size: [bold]{batch_size}[/bold]\n"
            f"· Normalization: [bold]{desc_normalize}[/bold]\n"
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
        **optimization_kwargs: Any,
    ) -> None:
        """Optimize reaction conditions using Bayesian Optimization.

        Args:
            batch_size: Number of conditions to recommend
            desc_normalize: Descriptor normalization method
            optimized_method: Optimization algorithm to use
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
        check_desc_completeness(self.desc_dict, self.condition_dict)
        self.total_name_arr, self.total_desc_arr = array_process(
            self.desc_dict, self.condition_dict, self.condition_types, desc_normalize, refine_desc
        )
        self.done_arr_index = done_array_process(self.prev_rxn_info, self.total_name_arr, self.condition_types)
        done_arr_desc = self.total_desc_arr[self.done_arr_index]
        done_arr_metrics = {k: self.prev_rxn_info[k].values for k in self.opt_metrics}

        # Normalize target values using opt_range
        self.y_scalers = {}
        normalized_metrics = {}
        for i, (metric, opt_metric) in enumerate(zip(self.opt_metrics, self.opt_metric_settings)):
            y_min, y_max = opt_metric["opt_range"]
            self.y_scalers[metric] = {"min": y_min, "max": y_max}
            normalized_y = (done_arr_metrics[metric] - y_min) / (y_max - y_min)  # Min-max normalization: (y - y_min) / (y_max - y_min)
            normalized_metrics[metric] = normalized_y

        device = torch.device(f"cuda:2") if torch.cuda.is_available() else torch.device("cpu")
        optimizer = Optimizer(
            name_data=self.total_name_arr,
            method=optimize_method,
            random_seed=self.random_seed,
            device=device,
            optimization_kwargs=optimization_kwargs,
        )

        self.selected_conditions, self.recommend_type, self.pred_mean, self.pred_std = optimizer.optimize(
            training_X=done_arr_desc,
            training_y=normalized_metrics,
            candidate_X=self.total_desc_arr,
            opt_metric_settings=self.opt_metric_settings,
            batch_size=batch_size,
        )

        # Denormalize prediction values using the same scalers
        if self.pred_mean is not None and self.pred_std is not None:
            denormalized_pred_mean = self.pred_mean.copy()
            denormalized_pred_std = self.pred_std.copy()

            for i, metric in enumerate(self.opt_metrics):
                y_min = self.y_scalers[metric]["min"]
                y_max = self.y_scalers[metric]["max"]
                denormalized_pred_mean[:, i] = self.pred_mean[:, i] * (y_max - y_min) + y_min
                denormalized_pred_std[:, i] = self.pred_std[:, i] * (y_max - y_min)

            # Update the attributes with denormalized values
            self.pred_mean = denormalized_pred_mean
            self.pred_std = denormalized_pred_std

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

    def save_results(
        self,
        save_dir: Union[str, Path],
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
        save_df(
            save_path=save_dir,
            filetype=filetype,
            selected_conditions=self.selected_conditions,
            condition_dict=self.condition_dict,
            recommend_type=self.recommend_type,
            batch_id=self.batch_id,
            pred_mean=self.pred_mean,
            pred_std=self.pred_std,
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
