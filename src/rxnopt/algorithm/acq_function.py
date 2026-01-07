import numpy as np
import gpytorch
import torch
from torch import Tensor
from botorch.models import ModelListGP
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.sampling.normal import SobolQMCNormalSampler

from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective

# from botorch.utils.multi_objective.scalarization import get_chebyshev_objective
from rxnopt.utils.logger import console


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

        self.acquisition_function = qLogExpectedHypervolumeImprovement(
            model=model,
            sampler=sampler,
            ref_point=ref_point,
            partitioning=partitioning,
        )

    @property
    def acq_func(self):
        return self.acquisition_function

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
    ) -> tuple[Tensor, Tensor]:
        acf_console = console
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
            available_choices = choices.clone()
            available_indices = torch.arange(len(choices), device=choices.device)

            for q_i in range(q):
                if len(available_choices) == 0:
                    acf_console.print(f"No more unique choices available for candidate {q_i+1}", style="red")
                    break

                progress.log(f"Chooseing candidate {q_i+1} of {q}", style="yellow")

                if unique:
                    keep_mask = torch.ones(len(available_choices), dtype=torch.bool, device=available_choices.device)

                    if exclude_points is not None:
                        for exclude_point in exclude_points:
                            distances = torch.norm(available_choices - exclude_point, dim=-1)
                            keep_mask = keep_mask & (distances > min_distance)

                    for selected_point in candidate_list:
                        distances = torch.norm(available_choices - selected_point.squeeze(), dim=-1)
                        keep_mask = keep_mask & (distances > min_distance)

                    available_choices = available_choices[keep_mask]
                    available_indices = available_indices[keep_mask]

                choices_batched = available_choices.unsqueeze(-2)
                with torch.no_grad():
                    with gpytorch.settings.cholesky_jitter(1e-3):
                        acq_values = self._split_batch_eval_acqf(
                            X=choices_batched,
                            max_batch_size=max_batch_size,
                            maximum_metrics=maximum_metrics,
                        )
                best_idx = torch.argmax(acq_values)
                selected_candidate = choices_batched[best_idx]

                candidate_list.append(selected_candidate)
                acq_value_list.append(acq_values[best_idx])

                candidates = torch.cat(candidate_list, dim=-2)
                torch.cuda.empty_cache()
                self.acquisition_function.set_X_pending(candidates)
                progress.update(task, advance=1)

            self.acquisition_function.set_X_pending(self.acquisition_function.X_pending)
            return candidates, torch.stack(acq_value_list)
        else:
            with torch.no_grad():
                acq_values = self._split_batch_eval_acqf(X=choices_batched, max_batch_size=max_batch_size, maximum_metrics=maximum_metrics)
            best_idx = torch.argmax(acq_values)
            return choices_batched[best_idx], acq_values[best_idx]

    def _split_batch_eval_acqf(self, X: Tensor, max_batch_size: int, maximum_metrics: bool = True) -> Tensor:
        acq_values_list = []
        with torch.no_grad():
            for X_batches in X.split(max_batch_size):
                acq_values = self.acquisition_function(X_batches)
                acq_values_list.append(acq_values)
        acq_values = torch.cat(acq_values_list, dim=0)
        return acq_values


class UCBAcquisitionFunction(BaseAcquisitionFunction):
    """
    Upper Confidence Bound acquisition function.
    对于多目标，通常会对合并后的标量化目标进行 UCB 计算。
    """

    def __init__(
        self,
        model: ModelListGP,
        sampler: SobolQMCNormalSampler,
        beta: float = 2.0,
        weights: Tensor = None,  # 用于标量化多目标的权重
    ):
        super().__init__(model, sampler)
        self.beta = beta

        # 如果提供了权重，则进行线性标量化
        objective = None
        if weights is not None:
            objective = GenericMCObjective(lambda Z, X: Z @ weights)
        self.acqusition_function = qUpperConfidenceBound(
            model=model,
            beta=beta,
            sampler=sampler,
            objective=objective,
        )

    @property
    def acq_func(self):
        return self.acqusition_function

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
    ) -> tuple[Tensor, Tensor]:
        acf_console = console
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
            available_choices = choices.clone()

            for q_i in range(q):
                if len(available_choices) == 0:
                    acf_console.print(f"No more unique choices available for candidate {q_i+1}", style="red")
                    break

                progress.log(f"Chooseing candidate {q_i+1} of {q}", style="yellow")

                if unique:
                    keep_mask = torch.ones(len(available_choices), dtype=torch.bool, device=available_choices.device)

                    if exclude_points is not None:
                        for exclude_point in exclude_points:
                            distances = torch.norm(available_choices - exclude_point, dim=-1)
                            keep_mask = keep_mask & (distances > min_distance)

                    for selected_point in candidate_list:
                        distances = torch.norm(available_choices - selected_point.squeeze(), dim=-1)
                        keep_mask = keep_mask & (distances > min_distance)

                    available_choices = available_choices[keep_mask]

                choices_batched = available_choices.unsqueeze(-2)
                with torch.no_grad():
                    acq_values = self._split_batch_eval_acqf(X=choices_batched, max_batch_size=max_batch_size)
                best_idx = torch.argmax(acq_values)
                selected_candidate = choices_batched[best_idx]

                candidate_list.append(selected_candidate)
                acq_value_list.append(acq_values[best_idx])

                candidates = torch.cat(candidate_list, dim=-2)
                torch.cuda.empty_cache()
                self.acqusition_function.set_X_pending(candidates)
                progress.update(task, advance=1)

            self.acqusition_function.set_X_pending(self.acqusition_function.X_pending)
            return candidates, torch.stack(acq_value_list)
        else:
            with torch.no_grad():
                acq_values = self._split_batch_eval_acqf(X=choices_batched, max_batch_size=max_batch_size)
            best_idx = torch.argmax(acq_values)
            return choices_batched[best_idx], acq_values[best_idx]

    def _split_batch_eval_acqf(self, X: Tensor, max_batch_size: int) -> Tensor:
        acq_values_list = []
        with torch.no_grad():
            for X_batches in X.split(max_batch_size):
                acq_values = self.acqusition_function(X_batches)
                acq_values_list.append(acq_values)
        acq_values = torch.cat(acq_values_list, dim=0)
        return acq_values


# ---------------------------------------------------------------------------
# 2. qParEGO (通过随机标量化解决多目标问题)
# ---------------------------------------------------------------------------
# class ParEGOAcquisitionFunction(BaseAcquisitionFunction):
#     """
#     ParEGO: 使用随机切比雪夫标量化将多目标转化为单目标，
#     然后应用 Expected Improvement (EI)。
#     """

#     def __init__(
#         self,
#         model: ModelListGP,
#         sampler: SobolQMCNormalSampler,
#         X_baseline: Tensor,  # 训练数据的输入特征
#         num_objectives: int,
#     ):
#         super().__init__(model, sampler)

#         # 1. 随机生成权重
#         weights = torch.randn(num_objectives).abs()
#         weights /= weights.sum()

#         # 2. 获取当前训练集的模型预测值用于标量化参考
#         with torch.no_grad():
#             posterior = model.posterior(X_baseline)
#             Y_baseline = posterior.mean

#         # 3. 构建切比雪夫标量化目标
#         # get_chebyshev_objective 会自动处理最大化/最小化
#         objective = get_chebyshev_objective(weights=weights, Y=Y_baseline)

#         # 4. 计算当前最佳标量化值
#         scalarized_Y = objective(Y_baseline)
#         best_f = scalarized_Y.max()
#         self.acq_func = qLogExpectedImprovement(
#             model=model,
#             best_f=best_f,
#             sampler=sampler,
#             objective=objective,
#         )

#     def __call__(self, X: torch.Tensor) -> torch.Tensor:
#         return self.acq_func(X)


# ---------------------------------------------------------------------------
# 3. qLogNEI (适用于有噪声的多目标观测)
# ---------------------------------------------------------------------------
class NEIAcquisitionFunction(BaseAcquisitionFunction):
    """
    Noisy Expected Improvement.
    当观测值存在显著噪声，或者无法直接确定 'best_f' 时效果极佳。
    """

    def __init__(
        self,
        model: ModelListGP,
        sampler: SobolQMCNormalSampler,
        X_baseline: Tensor,  # 必须提供已有的观察点
    ):
        super().__init__(model, sampler)

        # 对于多目标 NEI，通常需要指定如何整合多个输出
        # 这里默认采用求和或者你可以传入 objective
        self.acq_func = qLogNoisyExpectedImprovement(
            model=model,
            X_baseline=X_baseline,
            sampler=sampler,
            # objective=... (可选)
        )

    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        return self.acq_func(X)


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
