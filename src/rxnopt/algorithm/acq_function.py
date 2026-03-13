import numpy as np
import gpytorch
import torch
from torch import Tensor
from botorch.models import ModelListGP
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.sampling.normal import SobolQMCNormalSampler

from botorch.acquisition.monte_carlo import qUpperConfidenceBound, qExpectedImprovement
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization

from rxnopt.utils.logger import console


class BaseAcquisitionFunction:
    """Base class for acquisition functions"""

    def __init__(self, model: ModelListGP, sampler: SobolQMCNormalSampler):
        self.model = model
        self.sampler = sampler
        self.acquisition_function = None
        self.console = console

    @property
    def acq_func(self):
        return self.acquisition_function

    def _split_batch_eval_acqf(self, acq_func, X: Tensor, max_batch_size: int) -> Tensor:
        """Helper to evaluate acquisition function in batches to avoid OOM."""
        acq_values_list = []
        with torch.no_grad():
            for X_batches in X.split(max_batch_size):
                acq_values = acq_func(X_batches)
                acq_values_list.append(acq_values)
        return torch.cat(acq_values_list, dim=0)

    def optimize_discrete(
        self,
        acq_func,
        q: int,
        choices: Tensor,
        max_batch_size: int = 128,
        unique: bool = True,
        progress: object = None,
        task: object = None,
        min_distance: float = 1e-6,
        exclude_points: Tensor = None,
        temperature: float = 0.0,
        constraint_mask: Tensor = None,
    ) -> tuple[Tensor, Tensor]:
        """
        General discrete space greedy sequence optimization

        Args:
            acq_func: Acquisition function to optimize
            q: Number of candidates to select
            choices: Candidate points tensor
            max_batch_size: Maximum batch size for evaluation
            unique: Whether to ensure unique selections
            progress: Progress bar object
            task: Progress task object
            min_distance: Minimum distance for uniqueness check
            exclude_points: Points to exclude from selection
            temperature: Temperature for exploration-exploitation trade-off
            constraint_mask: Boolean mask indicating which points satisfy constraints
        """
        acq_func.set_X_pending(None)

        len_choices = len(choices)
        if len_choices < q and unique:
            self.console.print(
                f"Requested {q=} candidates but only {len_choices} choices remain.",
                style="yellow",
            )
            q = len_choices
        
        # Build the initial Available Mask
        available_mask = torch.ones(len_choices, dtype=torch.bool, device=choices.device)

        # Apply constraint mask if provided
        if constraint_mask is not None:
            available_mask = available_mask & constraint_mask
            constrained_count = constraint_mask.sum().item()
            total_count = len(constraint_mask)
            self.console.print(
                f"Constraint applied: {constrained_count}/{total_count} candidates available",
                style="cyan"
            )

        # Exclude the specified points
        if unique and exclude_points is not None and len(exclude_points) > 0:
            dists = torch.cdist(choices, exclude_points)
            distinct_check = (dists > min_distance).all(dim=1)
            available_mask = available_mask & distinct_check
        candidate_list = []
        acq_value_list = []

        current_mask = available_mask.clone()
        # greedy get loop
        for q_i in range(q):
            valid_indices = torch.nonzero(current_mask, as_tuple=True)[0]
            if len(valid_indices) == 0:
                self.console.print(f"No more unique choices available for candidate {q_i+1}", style="red")
                break

            if progress:
                progress.log(f"Choosing candidate {q_i+1} of {q}", style="yellow")

            choices_batched = choices[valid_indices].unsqueeze(-2)

            with torch.no_grad():

                with gpytorch.settings.cholesky_jitter(1e-3):
                    acq_values = self._split_batch_eval_acqf(
                        acq_func=acq_func,
                        X=choices_batched,
                        max_batch_size=max_batch_size,
                    )

            # define the exploration and exploitation trade-off
            if temperature > 0.0:
                range_val = (torch.max(acq_values) - torch.min(acq_values)).clamp(min=1e-8)
                acq_norm = (acq_values - torch.min(acq_values)) / range_val
                logits = (acq_norm - 1.0) / temperature
                probs = torch.softmax(logits, dim=0)

                best_idx_in_batch = torch.multinomial(probs, 1).item()
            else:
                best_idx_in_batch = torch.argmax(acq_values)

            best_acq_val = acq_values[best_idx_in_batch]

            best_global_idx = valid_indices[best_idx_in_batch]

            # (1, 1, D)
            selected_candidate = choices[best_global_idx].unsqueeze(0).unsqueeze(0)

            candidate_list.append(selected_candidate)
            acq_value_list.append(best_acq_val)

            current_candidates = torch.cat(candidate_list, dim=-2)
            torch.cuda.empty_cache()

            # Key step: Inform the acquisition function that these points have been selected.
            # When looking for the next point, it should be based on the conditional distribution of these points
            acq_func.set_X_pending(current_candidates.squeeze(0))
            # If uniqueness is required, update the mask to exclude the point just selected
            if unique:
                dist_to_new = torch.norm(choices - selected_candidate.squeeze(), dim=-1)
                is_far_enough = dist_to_new > min_distance
                current_mask = current_mask & is_far_enough
            if progress:
                progress.update(task, advance=1)

        # clean up
        acq_func.set_X_pending(None)

        if not candidate_list:
            return torch.empty((0, choices.shape[-1]), device=choices.device), torch.empty(0, device=choices.device)

        final_candidates = torch.cat(candidate_list, dim=-2).squeeze(0)  # (q, D)
        final_values = torch.stack(acq_value_list)

        return final_candidates, final_values

    def optimize_acqf_discrete(self, *args, **kwargs):
        """Evaluate the acquisition function"""
        raise NotImplementedError


class EHVIAcquisitionFunction(BaseAcquisitionFunction):
    """Enhanced Expected Hypervolume Improvement acquisition function"""

    def __init__(
        self,
        model: ModelListGP,
        sampler: SobolQMCNormalSampler,
        ref_point: torch.Tensor,
        partitioning,
    ):
        super().__init__(model, sampler)
        self.ref_point = ref_point
        self.partitioning = partitioning
        self.acquisition_function = qLogExpectedHypervolumeImprovement(
            model=model,
            sampler=sampler,
            ref_point=ref_point,
            partitioning=partitioning,
        )

    def optimize_acqf_discrete(
        self,
        q: int,
        choices: Tensor,
        max_batch_size: int = 128,
        unique: bool = True,
        progress: object = None,
        task: object = None,
        min_distance: float = 1e-6,
        exclude_points: Tensor = None,
        temperature: float = 0.0,
        constraint_mask: Tensor = None,
    ) -> tuple[Tensor, Tensor]:

        return self.optimize_discrete(
            acq_func=self.acquisition_function,
            q=q,
            choices=choices,
            max_batch_size=max_batch_size,
            unique=unique,
            progress=progress,
            task=task,
            min_distance=min_distance,
            exclude_points=exclude_points,
            temperature=temperature,
            constraint_mask=constraint_mask,
        )


class UCBAcquisitionFunction(BaseAcquisitionFunction):
    """
    Upper Confidence Bound acquisition function.
    """

    def __init__(
        self,
        model: ModelListGP,
        sampler: SobolQMCNormalSampler,
        beta: float = 2.0,
        weights: Tensor = None,
    ):
        super().__init__(model, sampler)
        self.beta = beta
        objective = None
        if weights is not None:
            objective = GenericMCObjective(lambda Z, X: Z @ weights)

        self.acquisition_function = qUpperConfidenceBound(
            model=model,
            beta=beta,
            sampler=sampler,
            objective=objective,
        )

    def optimize_acqf_discrete(
        self,
        q: int,
        choices: Tensor,
        max_batch_size: int = 128,
        unique: bool = True,
        maximum_metrics: bool = True,
        progress: object = None,
        task: object = None,
        min_distance: float = 1e-6,
        exclude_points: Tensor = None,
        temperature: float = 0.0,
        constraint_mask: Tensor = None,
    ) -> tuple[Tensor, Tensor]:
        # 直接调用基类的通用逻辑
        return self.optimize_discrete(
            acq_func=self.acquisition_function,
            q=q,
            choices=choices,
            max_batch_size=max_batch_size,
            unique=unique,
            progress=progress,
            task=task,
            min_distance=min_distance,
            exclude_points=exclude_points,
            temperature=temperature,
            constraint_mask=constraint_mask,
        )


class ParEGOAcquisitionFunction(BaseAcquisitionFunction):
    """
    Convert multiple objectives into single objectives using random Chebyshev scaling, and then apply EI.
    """

    def __init__(
        self,
        model: ModelListGP,
        sampler: SobolQMCNormalSampler,
        X_baseline: Tensor,
        num_objectives: int,
    ):
        super().__init__(model, sampler)

        weights = torch.randn(num_objectives).abs()
        weights /= weights.sum()

        with torch.no_grad():
            posterior = model.posterior(X_baseline)
            Y_baseline = posterior.mean

        weights = weights.to(Y_baseline.device)

        objective = self._get_chebyshev_objective(weights=weights, Y=Y_baseline)

        scalarized_Y = objective(Y_baseline)
        best_f = scalarized_Y.max()

        self.acquisition_function = qExpectedImprovement(
            model=model,
            best_f=best_f,
            sampler=sampler,
            objective=objective,
        )

    def optimize_acqf_discrete(
        self,
        q: int,
        choices: Tensor,
        max_batch_size: int = 128,
        unique: bool = True,
        maximum_metrics: bool = True,
        progress: object = None,
        task: object = None,
        min_distance: float = 1e-6,
        exclude_points: Tensor = None,
        temperature: float = 0.0,
        constraint_mask: Tensor = None,
    ) -> tuple[Tensor, Tensor]:
        return self.optimize_discrete(
            acq_func=self.acquisition_function,
            q=q,
            choices=choices,
            max_batch_size=max_batch_size,
            unique=unique,
            progress=progress,
            task=task,
            min_distance=min_distance,
            exclude_points=exclude_points,
            temperature=temperature,
            constraint_mask=constraint_mask,
        )

    @staticmethod
    def _get_chebyshev_objective(weights: Tensor, Y: Tensor) -> GenericMCObjective:
        """
        创建一个 GenericMCObjective，它对输出应用切比雪夫标量化。

        Args:
            weights: 权重向量 (num_objectives,)
            Y:以此为基准计算理想点(ideal point)和最差点(nadir point)的观测值 (n x num_objectives)
        """
        # get_chebyshev_scalarization 返回一个 python callable (function)
        # 这个 callable 接受 tensor 并返回标量化后的 tensor
        scalarization_fn = get_chebyshev_scalarization(weights=weights, Y=Y)

        # 将其包装为 BoTorch 采集函数可用的 MCObjective 对象
        return GenericMCObjective(scalarization_fn)


# ---------------------------------------------------------------------------
# 3. qLogNEI (适用于有噪声的多目标观测)
# ---------------------------------------------------------------------------
class NEIAcquisitionFunction(BaseAcquisitionFunction):
    """
    Noisy Expected Improvement (Log-space version).
    当观测值存在显著噪声，或者无法直接确定 'best_f' 时效果极佳。
    使用 LogNoisyExpectedImprovement 以获得更好的数值稳定性。
    """

    def __init__(
        self,
        model: ModelListGP,
        sampler: SobolQMCNormalSampler,
        X_baseline: Tensor,  # 必须提供已有的观察点
        weights: Tensor = None,
        prune_baseline: bool = True,  # 这是一个通常有用的选项，用于减少计算量
    ):
        super().__init__(model, sampler)

        # 处理多目标/多输出的权重聚合
        objective = None
        if weights is not None:
            objective = GenericMCObjective(lambda Z, X: Z @ weights)
        # 初始化 BoTorch 的 qLogNoisyExpectedImprovement
        self.acquisition_function = qLogNoisyExpectedImprovement(
            model=model,
            X_baseline=X_baseline,
            sampler=sampler,
            objective=objective,
            prune_baseline=prune_baseline,
        )

    def optimize_acqf_discrete(
        self,
        q: int,
        choices: Tensor,
        max_batch_size: int = 128,
        unique: bool = True,
        maximum_metrics: bool = True,
        progress: object = None,
        task: object = None,
        min_distance: float = 1e-6,
        exclude_points: Tensor = None,
        temperature: float = 0.0,
        constraint_mask: Tensor = None,
    ) -> tuple[Tensor, Tensor]:
        # 直接调用基类的通用贪婪优化逻辑
        return self.optimize_discrete(
            acq_func=self.acquisition_function,
            q=q,
            choices=choices,
            max_batch_size=max_batch_size,
            unique=unique,
            progress=progress,
            task=task,
            min_distance=min_distance,
            exclude_points=exclude_points,
            temperature=temperature,
            constraint_mask=constraint_mask,
        )


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
