import numpy as np
import gpytorch
import torch
from torch import Tensor
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models import ModelListGP
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.sampling.normal import SobolQMCNormalSampler

from rich.console import Console


def optimize_acqf_discrete(
    acq_function: AcquisitionFunction,
    q: int,
    choices: Tensor,
    max_batch_size: int = 128,
    unique: bool = True,
    maximum_metrics: bool = True,
    progress: object = None,
    task: object = None,
    min_distance: float = 1e-6,
    exclude_points: Tensor = None,
) -> tuple[Tensor, Tensor]:
    r"""Optimize over a discrete set of points using batch evaluation.

    For `q > 1` this function generates candidates by means of sequential
    conditioning (rather than joint optimization), since for all but the
    smalles number of choices the set `choices^q` of discrete points to
    evaluate quickly explodes.

    Args:
        acq_function: An AcquisitionFunction.
        q: The number of candidates.
        choices: A `num_choices x d` tensor of possible choices.
        max_batch_size: The maximum number of choices to evaluate in batch.
            A large limit can cause excessive memory usage if the model has
            a large training set.
        unique: If True return unique choices, o/w choices may be repeated
            (only relevant if `q > 1`).
        min_distance: Minimum distance between selected points when unique=True.
        exclude_points: Tensor of points to avoid (e.g., training data).

    Returns:
        A two-element tuple containing

        - a `q x d`-dim tensor of generated candidates.
        - an associated acquisition value.
    """
    acf_console = Console()
    len_choices = len(choices)
    if len_choices < q and unique:
        acf_console.print(
            f"Requested {q=} candidates from fully discrete search space, but only {len_choices} possible choices remain. ",
            style="yellow",
        )
        q = len_choices
    choices_batched = choices.unsqueeze(-2)

    if q > 1:
        candidate_list, acq_value_list = [], []
        available_choices = choices.clone()  # 创建候选点的副本
        available_indices = torch.arange(len(choices), device=choices.device)  # 追踪可用候选点的索引

        for q_i in range(q):
            if len(available_choices) == 0:
                acf_console.print(f"No more unique choices available for candidate {q_i+1}", style="red")
                break

            progress.log(f"Chooseing candidate {q_i+1} of {q}", style="yellow")

            if unique:
                keep_mask = torch.ones(len(available_choices), dtype=torch.bool, device=available_choices.device)

                # 检查与所有历史点（训练数据等）的距离
                if exclude_points is not None:
                    for exclude_point in exclude_points:
                        distances = torch.norm(available_choices - exclude_point, dim=-1)
                        keep_mask = keep_mask & (distances > min_distance)

                # 检查与当前批次中已选择点的距离
                for selected_point in candidate_list:
                    distances = torch.norm(available_choices - selected_point.squeeze(), dim=-1)
                    keep_mask = keep_mask & (distances > min_distance)

                available_choices = available_choices[keep_mask]
                available_indices = available_indices[keep_mask]

            choices_batched = available_choices.unsqueeze(-2)
            with torch.no_grad():
                with gpytorch.settings.cholesky_jitter(1e-3):
                    acq_values = _split_batch_eval_acqf(
                        acq_function=acq_function,
                        X=choices_batched,
                        max_batch_size=max_batch_size,
                        maximum_metrics=maximum_metrics,
                    )
            best_idx = torch.argmax(acq_values)
            selected_candidate = choices_batched[best_idx]

            candidate_list.append(selected_candidate)
            acq_value_list.append(acq_values[best_idx])

            # 设置 pending points
            candidates = torch.cat(candidate_list, dim=-2)
            torch.cuda.empty_cache()  # 清空缓存（可选）
            acq_function.set_X_pending(candidates)
            progress.update(task, advance=1)

        # Reset acq_func to previous X_pending state
        acq_function.set_X_pending(acq_function.X_pending)
        return candidates, torch.stack(acq_value_list)
    else:
        with torch.no_grad():
            acq_values = _split_batch_eval_acqf(
                acq_function=acq_function, X=choices_batched, max_batch_size=max_batch_size, maximum_metrics=maximum_metrics
            )
        best_idx = torch.argmax(acq_values)
        return choices_batched[best_idx], acq_values[best_idx]


def _split_batch_eval_acqf(acq_function: AcquisitionFunction, X: Tensor, max_batch_size: int, maximum_metrics: bool) -> Tensor:

    acq_values_list = []
    with torch.no_grad():
        for X_batches in X.split(max_batch_size):
            acq_values = acq_function(X_batches)
            acq_values_list.append(acq_values)
    acq_values = torch.cat(acq_values_list, dim=0)
    return acq_values


class BaseAcquisitionFunction:
    """Base class for acquisition functions"""

    def __init__(self, model: ModelListGP, sampler: SobolQMCNormalSampler):
        self.model = model
        self.sampler = sampler

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the acquisition function"""
        raise NotImplementedError


class EHVIAcquisitionFunction(BaseAcquisitionFunction):
    """Enhanced Expected Hypervolume Improvement acquisition function with exploration bonus"""

    def __init__(
        self,
        model: ModelListGP,
        sampler: SobolQMCNormalSampler,
        ref_point: torch.Tensor,
        partitioning: NondominatedPartitioning,
        exploration_weight: float = 0.1,  # New parameter for exploration
    ):
        super().__init__(model, sampler)
        self.ref_point = ref_point
        self.partitioning = partitioning
        self.exploration_weight = exploration_weight

        self.ehvi = qLogExpectedHypervolumeImprovement(
            model=model,
            sampler=sampler,
            ref_point=ref_point,
            partitioning=partitioning,
        )

    def evaluate_with_exploration(self, X: torch.Tensor) -> torch.Tensor:
        """Enhanced evaluation with exploration bonus"""
        # Standard EHVI
        ehvi_val = self.ehvi(X)

        # Add exploration bonus based on predictive variance
        with torch.no_grad():
            posterior = self.model.posterior(X)
            # Average variance across objectives as exploration signal
            exploration_bonus = torch.mean(posterior.variance, dim=-1)

        return ehvi_val + self.exploration_weight * exploration_bonus


class ParetoFrontCalculator:
    """Class for calculating Pareto fronts"""

    @staticmethod
    def calculate_target_function(points: np.ndarray, progress: object, task: object) -> np.ndarray:
        """
        Calculate Pareto front for points in arbitrary dimensions

        Args:
            points: numpy array of shape (n_points, n_dimensions)

        Returns:
            numpy array of Pareto optimal points
        """
        if len(points) == 0:
            return np.array([])
        pareto_front = [points[0]]  # Initialize list of Pareto optimal points
        for point in points[1:]:
            progress.update(task, advance=1)
            is_pareto = True
            to_remove = []
            # Compare with all points in current Pareto front
            for i, pf_point in enumerate(pareto_front):
                # Check if the current point dominates any existing Pareto point
                if np.all(point >= pf_point) and np.any(point > pf_point):
                    to_remove.append(i)
                # Check if any existing Pareto point dominates the current point
                elif np.all(point <= pf_point) and np.any(point < pf_point):
                    is_pareto = False
                    break

            # Remove dominated points from Pareto front
            for i in reversed(to_remove):
                pareto_front.pop(i)

            # Add current point if it's Pareto optimal
            if is_pareto:
                pareto_front.append(point)
        return torch.tensor(np.array(pareto_front))
