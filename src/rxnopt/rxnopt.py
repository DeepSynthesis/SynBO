"""Reaction Optimization Framework.

A modern, efficient framework for multi-objective reaction optimization
using Bayesian Optimization techniques.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import pandas as pd
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .optimize import Optimizer
from .utils.utils import (
    check_desc_completeness,
    done_array_process,
    generate_onehot_desc,
    track_called,
    array_process,
)
from .initialize import Initializer
from .utils.write_excel import ExcelWriter

console = Console()


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

    def __init__(self, opt_metrics: Union[str, List[str]], opt_type: Literal["init", "opt", "auto"] = "auto") -> None:
        if isinstance(opt_metrics, str):
            opt_metrics = [opt_metrics]
        elif not isinstance(opt_metrics, list):
            raise ValueError("opt_metrics must be str or list")

        if opt_type not in ["init", "opt", "auto"]:
            raise ValueError("opt_type must be 'init', 'opt' or 'auto'")

        self.condition_dict: Dict[str, List[Any]] = {}
        self.desc_dict: Dict[str, Any] = {}
        self.opt_metrics = opt_metrics
        self.opt_type = opt_type
        self.prev_rxn_info: Optional[pd.DataFrame] = None
        self.batch_id = 0

        console.print(
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
        condition_dict = {k: sorted(v) for k, v in sorted(condition_dict.items(), key=lambda x: x[0])}
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

        console.print(table)

    def load_desc(self, desc_dict: Optional[Dict[str, Any]] = None) -> None:
        """Load descriptor dictionary.

        Args:
            desc_dict: Optional descriptor dictionary. If None, uses OneHot encoding.

        Raises:
            AssertionError: If condition types don't match
        """
        if desc_dict is None:
            console.print("[yellow]Warning: No descriptor dictionary provided, " "using OneHot encoding as alternative.[/yellow]")
            self.desc_dict = generate_onehot_desc(self.condition_dict)
        else:
            if set(desc_dict.keys()) != set(self.condition_types):
                raise ValueError("Condition types do not match")
            self.desc_dict = desc_dict

        console.print("[green]✓ Descriptors loaded successfully[/green]")

    @track_called
    def load_prev_rxn(self, prev_rxn_info: pd.DataFrame, drop_rxn: bool = False) -> None:
        """Load previous reaction information.

        Args:
            prev_rxn_info: DataFrame containing previous reaction data
            drop_rxn: Whether to drop reactions with missing species

        Raises:
            ValueError: If species not found in condition space
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Loading previous reactions...", total=None)

            self.opt_type = "opt" if self.opt_type == "auto" else self.opt_type
            self.batch_id = prev_rxn_info["batch"].max() + 1

            # Validate condition types
            missing_types = [t for t in self.condition_types if t not in prev_rxn_info.columns]
            if missing_types:
                raise ValueError(f"Missing condition types: {missing_types}")

            # Check for missing species in each condition type
            for condition_type in self.condition_types:
                progress.update(task, description=f"Validating {condition_type}...")

                missing_species = set(prev_rxn_info[condition_type]) - set(self.condition_dict[condition_type])

                if missing_species:
                    if drop_rxn:
                        console.print(
                            f"[yellow]Warning: {missing_species} not in {condition_type} "
                            f"condition space, dropping these reactions[/yellow]"
                        )
                        prev_rxn_info = prev_rxn_info[~prev_rxn_info[condition_type].isin(missing_species)]
                    else:
                        raise ValueError(f"{missing_species} not in {condition_type} condition space")

            # Convert metrics to float
            for opt_metric in self.opt_metrics:
                prev_rxn_info[opt_metric] = prev_rxn_info[opt_metric].astype(float)

            self.prev_rxn_info = prev_rxn_info
            progress.update(task, description="Previous reactions loaded!")

        console.print(f"[green]✓ Loaded {len(prev_rxn_info)} previous reactions[/green]")

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
            console.print("[yellow]Reaction space expansion not yet implemented[/yellow]")

        console.print(
            Panel(
                f"[bold]Running in {self.opt_type.upper()} mode[/bold]\n" f"Batch size: {batch_size}\n" f"Normalization: {desc_normalize}",
                title="🚀 Execution Plan",
            )
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
        sampling_method: Literal["sobol", "random", "lhs"] = "sobol",
    ) -> None:
        """Initialize reaction optimization with initial sampling.

        Args:
            batch_size: Number of initial samples
            desc_normalize: Descriptor normalization method
            sampling_method: Sampling strategy for initial points
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Initializing...", total=None)

            progress.update(task, description="Checking descriptor completeness...")
            check_desc_completeness(self.desc_dict, self.condition_dict)

            progress.update(task, description="Processing arrays...")
            self.total_name_arr, self.total_desc_arr = array_process(
                self.desc_dict, self.condition_dict, self.condition_types, desc_normalize
            )

            progress.update(task, description="Selecting initial points...")
            initializer = Initializer(numerical_data=self.total_desc_arr, name_data=self.total_name_arr)
            self.selected_conditions = initializer.sampling(method=sampling_method, batch_size=batch_size)

            # All initial points are exploration
            self.recommend_type = ["explore"] * batch_size
            progress.update(task, description="Initialization complete!")

        console.print(f"[green]✓ Selected {batch_size} initial conditions using " f"{sampling_method} sampling[/green]")

    def optimize(
        self,
        batch_size: int = 5,
        desc_normalize: Literal["minmax", "zscore", "l2"] = "minmax",
        optimized_method: str = "default",
        opt_weights: Optional[List[float]] = None,
        mc_num_samples: int = 128,
        max_batch_size: int = 128,
        gpu_id: int = 0,
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
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Optimizing...", total=None)

            progress.update(task, description="Checking descriptor completeness...")
            check_desc_completeness(self.desc_dict, self.condition_dict)

            progress.update(task, description="Processing arrays...")
            self.total_name_arr, self.total_desc_arr = array_process(
                self.desc_dict, self.condition_dict, self.condition_types, desc_normalize
            )

            progress.update(task, description="Processing historical data...")
            self.done_arr_index = done_array_process(self.prev_rxn_info, self.total_name_arr, self.condition_types)
            done_arr_desc = self.total_desc_arr[self.done_arr_index]
            done_arr_metrics = {k: self.prev_rxn_info[k].values for k in self.opt_metrics}

            progress.update(task, description="Running Bayesian optimization...")
            optimizer = Optimizer(
                name_data=self.total_name_arr,
                method=optimized_method,
                mc_num_samples=mc_num_samples,
                max_batch_size=max_batch_size,
            )

            self.selected_conditions, self.recommend_type = optimizer.optimize(
                training_X=done_arr_desc,
                training_y=done_arr_metrics,
                candidate_X=self.total_desc_arr,
                batch_size=batch_size,
                opt_weights=opt_weights,
                gpu_id=gpu_id,
            )

            progress.update(task, description="Optimization complete!")

        # Display optimization summary
        exploit_count = sum(1 for t in self.recommend_type if t == "exploit")
        explore_count = sum(1 for t in self.recommend_type if t == "explore")

        console.print(
            Panel(
                f"[green]Optimization Complete![/green]\n"
                f"Recommended: {batch_size} conditions\n"
                f"Exploit: {exploit_count} | Explore: {explore_count}\n"
                f"Method: {optimized_method} | GPU: {gpu_id}",
                title="🎯 Results Summary",
            )
        )

    def save_recommendations(
        self,
        save_task: Union[str, Path],
        filetype: Literal["csv", "excel"] = "csv",
        figure_output: List[str] = None,
        figure_path: Optional[Union[str, Path]] = None,
        suffix: Optional[str] = None,
    ) -> None:
        """Save recommendations to file.

        Args:
            save_task: Directory path to save results
            filetype: Output file format
            figure_output: List of figure types to generate
            figure_path: Path for figures
            suffix: Optional filename suffix
        """
        if figure_output is None:
            figure_output = []

        file_name = f"batch-{self.batch_id}_{datetime.now().strftime('%Y%m%d')}"
        if suffix:
            file_name = f"{file_name}_{suffix}"

        save_path = Path(save_task) / file_name

        # Create directory if it doesn't exist
        if not save_path.parent.exists():
            console.print(f"[yellow]Creating directory: {save_path.parent}[/yellow]")
            save_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare output DataFrame
        output_df = pd.DataFrame(
            {
                "batch": [self.batch_id] * len(self.selected_conditions),
                "index": range(1, len(self.selected_conditions) + 1),
                "type": self.recommend_type,
                **pd.DataFrame(self.selected_conditions, columns=self.condition_types).to_dict("list"),
                **{metric: "" for metric in self.opt_metrics},
            }
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Saving recommendations...", total=None)

            if filetype == "csv":
                progress.update(task, description="Writing CSV file...")
                output_df.to_csv(save_path.with_suffix(".csv"), index=False)
            elif filetype == "excel":
                progress.update(task, description="Writing Excel file...")
                writer = ExcelWriter(condition_types=self.condition_types, opt_metrics=self.opt_metrics)
                writer.write_to_excel(
                    output_df=output_df,
                    batch_id=self.batch_id,
                    figure_output=figure_output,
                    figure_path=figure_path,
                    save_path=save_path,
                )
            else:
                raise ValueError(f"Unknown filetype: {filetype}")

        console.print(f"[green]✓ Saved recommendations to: {save_path.with_suffix('.' + filetype)}[/green]")
