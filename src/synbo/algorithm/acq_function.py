import numpy as np
import gpytorch
import torch
from torch import Tensor
from botorch.models import ModelListGP
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement, qLogExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.sampling.normal import SobolQMCNormalSampler

from botorch.acquisition.monte_carlo import qUpperConfidenceBound, qExpectedImprovement
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.optim import optimize_acqf_discrete

from synbo.utils.logger import console


class BaseAcquisitionFunction:
    """Base class for acquisition functions"""

    def __init__(self, model: ModelListGP, sampler: SobolQMCNormalSampler, device: torch.device):
        self.model = model
        self.sampler = sampler
        self.acquisition_function = None
        self.console = console
        self.device = device

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
        unique: bool = True,
        max_batch_size: int = 1024,
        progress: object = None,
        task: object = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Joint optimization for batch selection (EDBO-style).
        This method selects q candidates simultaneously using BoTorch's optimize_acqf_discrete,
        exactly as implemented in EDBO+.

        Unlike the greedy sequence optimization, this approach considers the joint expected improvement
        of all q candidates at once. This is the implementation used by EDBO+.

        Args:
            acq_func: Acquisition function to optimize (must support q-batch evaluation)
            q: Number of candidates to select
            choices: Candidate points tensor (n, D)
            unique: Whether to ensure unique selections (default: True)
            max_batch_size: Maximum batch size for evaluation (default: 128)
            progress: Not used in EDBO implementation (kept for API compatibility)
            task: Not used in EDBO implementation (kept for API compatibility)

        Returns:
            tuple[Tensor, Tensor]:
                - Selected candidates (q, D)
                - Acquisition values for selected candidates (q,)
        """

        acq_result = self._new_optimize_acqf_discrete(
            acq_function=acq_func,
            choices=choices,
            q=q,
            unique=unique,
            max_batch_size=max_batch_size,
            progress=progress,
            task=task,
        )

        # acq_result is a tuple of (candidates, acquisition_values)

        selected_candidates = acq_result[0]  # (q, D)
        acquisition_values = acq_result[1]  # (q,)

        return selected_candidates, acquisition_values

    def _new_optimize_acqf_discrete(
        self,
        acq_function,
        q,
        choices,
        max_batch_size,
        unique,
        progress: object = None,
        task: object = None,
    ):

        def _split_batch_eval_acqf(acq_function, X, max_batch_size):
            acq_values_list = []
            for X_batch in X.split(max_batch_size):
                X_batch_gpu = X_batch.to(device=self.device)
                acq_batch = acq_function(X_batch_gpu)
                acq_values_list.append(acq_batch.cpu())
                del X_batch_gpu
            return torch.cat(acq_values_list, dim=0)

        choices_batched = choices.unsqueeze(-2)
        if q > 1:
            candidate_list, acq_value_list = [], []
            for q_i in range(q):
                if progress and task:
                    progress.update(task, advance=1)
                with torch.no_grad():
                    acq_values = _split_batch_eval_acqf(
                        acq_function=acq_function,
                        X=choices_batched,
                        max_batch_size=max_batch_size,
                    )

                best_idx = torch.argmax(acq_values)

                candidate_list.append(choices_batched[best_idx])
                acq_value_list.append(acq_values[best_idx])
                candidates = torch.cat(candidate_list, dim=-2)

                acq_function.set_X_pending(candidates.to(self.device))

                if unique:
                    choices_batched = torch.cat([choices_batched[:best_idx], choices_batched[best_idx + 1 :]])

            return candidates, torch.stack(acq_value_list)

        with torch.no_grad():
            acq_values = _split_batch_eval_acqf(acq_function=acq_function, X=choices_batched, max_batch_size=max_batch_size)

        best_idx = torch.argmax(acq_values)
        return choices_batched[best_idx], acq_values[best_idx]


class EHVIAcquisitionFunction(BaseAcquisitionFunction):
    """Enhanced Expected Hypervolume Improvement acquisition function"""

    def __init__(
        self,
        model: ModelListGP,
        sampler: SobolQMCNormalSampler,
        ref_point: torch.Tensor,
        partitioning,
        train_x,
        device,
    ):
        super().__init__(model, sampler, device)
        self.ref_point = ref_point
        self.partitioning = partitioning
        # self.acquisition_function = qLogNoisyExpectedHypervolumeImprovement(
        #     model=model,
        #     sampler=sampler,
        #     ref_point=ref_point,
        #     alpha=0.0,
        #     incremental_nehvi=True,
        #     X_baseline=train_x,
        #     prune_baseline=True,
        # )
        self.acquisition_function = qLogNoisyExpectedImprovement(
            model=model,
            sampler=sampler,
            # ref_point=ref_point,
            # alpha=0.0,
            # incremental_nehvi=True,
            X_baseline=train_x,
            # prune_baseline=True,
        )

    def optimize_acqf_discrete(
        self,
        q: int,
        choices: Tensor,
        max_batch_size: int = 1024,
        unique: bool = True,
        progress: object = None,
        task: object = None,
    ) -> tuple[Tensor, Tensor]:

        return self.optimize_discrete(
            acq_func=self.acquisition_function,
            q=q,
            choices=choices,
            max_batch_size=max_batch_size,
            unique=unique,
            progress=progress,
            task=task,
        )


class UCBAcquisitionFunction(BaseAcquisitionFunction):
    """
    Upper Confidence Bound acquisition function.
    """

    def __init__(
        self,
        model: ModelListGP,
        sampler: SobolQMCNormalSampler,
        device,
        beta: float = 2.0,
        weights: Tensor = None,
    ):
        super().__init__(model, sampler, device)
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
        max_batch_size: int = 1024,
        unique: bool = True,
        progress: object = None,
        task: object = None,
    ) -> tuple[Tensor, Tensor]:
        # Directly call base class general logic
        return self.optimize_discrete(
            acq_func=self.acquisition_function,
            q=q,
            choices=choices,
            max_batch_size=max_batch_size,
            unique=unique,
            progress=progress,
            task=task,
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
        device,
    ):
        super().__init__(model, sampler, device)

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
        max_batch_size: int = 1024,
        unique: bool = True,
        maximum_metrics: bool = True,
        progress: object = None,
        task: object = None,
    ) -> tuple[Tensor, Tensor]:
        return self.optimize_discrete(
            acq_func=self.acquisition_function,
            q=q,
            choices=choices,
            max_batch_size=max_batch_size,
            unique=unique,
            progress=progress,
            task=task,
        )

    @staticmethod
    def _get_chebyshev_objective(weights: Tensor, Y: Tensor) -> GenericMCObjective:
        """
        Create a GenericMCObjective that applies Chebyshev scalarization to outputs.

        Args:
            weights: Weight vector (num_objectives,)
            Y: Observations to calculate ideal point and nadir point (n x num_objectives)
        """
        # get_chebyshev_scalarization returns a python callable (function)
        # This callable accepts tensor and returns scalarized tensor
        scalarization_fn = get_chebyshev_scalarization(weights=weights, Y=Y)

        # Wrap it as an MCObjective object usable by BoTorch acquisition functions
        return GenericMCObjective(scalarization_fn)


class NEIAcquisitionFunction(BaseAcquisitionFunction):
    """
    Noisy Expected Improvement (Log-space version).
    Works excellently when observations have significant noise or when 'best_f' can't be determined.
    Use LogNoisyExpectedImprovement for better numerical stability。
    """

    def __init__(
        self,
        model: ModelListGP,
        sampler: SobolQMCNormalSampler,
        X_baseline: Tensor,  # Must provide existing observation points
        device,
        weights: Tensor = None,
        prune_baseline: bool = True,  # This is a generally useful option，to reduce computation
    ):
        super().__init__(model, sampler, device)

        # Handle weight aggregation for multi-objective/multi-output
        objective = None
        if weights is not None:
            objective = GenericMCObjective(lambda Z, X: Z @ weights)
        # Initialize BoTorch
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
        max_batch_size: int = 1024,
        unique: bool = True,
        maximum_metrics: bool = True,
        progress: object = None,
        task: object = None,
    ) -> tuple[Tensor, Tensor]:
        # Directly call base class general greedy optimization logic
        return self.optimize_discrete(
            acq_func=self.acquisition_function,
            q=q,
            choices=choices,
            max_batch_size=max_batch_size,
            unique=unique,
            progress=progress,
            task=task,
        )


class ParetoFrontCalculator:
    """Class for calculating Pareto fronts"""

    @staticmethod
    def calculate_target_function(points: np.ndarray, progress: object, task: object) -> np.ndarray:
        # def calculate_target_function(points: np.ndarray) -> np.ndarray:
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
            if progress and task:
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
