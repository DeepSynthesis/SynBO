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
        self.acquisition_function = None  # 子类应实例化此对象
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

    def optimize_discrete_greedy(
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
    ) -> tuple[Tensor, Tensor]:
        """
        通用的离散空间贪婪序列优化逻辑。
        适用于 qUCB, qEHVI, qNEHVI 等支持 set_X_pending 的采集函数。
        """
        acq_func.set_X_pending(None)

        len_choices = len(choices)
        if len_choices < q and unique:
            self.console.print(
                f"Requested {q=} candidates but only {len_choices} choices remain.",
                style="yellow",
            )
            q = len_choices
        # 2. 构建初始可用掩码 (Available Mask)
        available_mask = torch.ones(len_choices, dtype=torch.bool, device=choices.device)

        # 排除指定的点 (exclude_points)
        if unique and exclude_points is not None and len(exclude_points) > 0:
            # 计算 choices 到 exclude_points 的距离矩阵
            dists = torch.cdist(choices, exclude_points)
            # 如果某点到任意 exclude_point 的距离小于阈值，则该点不可用
            distinct_check = (dists > min_distance).all(dim=1)
            available_mask = available_mask & distinct_check
        candidate_list = []
        acq_value_list = []
        # 当前迭代的掩码
        current_mask = available_mask.clone()
        # 3. 贪婪循环
        for q_i in range(q):
            # 获取当前有效索引
            valid_indices = torch.nonzero(current_mask, as_tuple=True)[0]
            if len(valid_indices) == 0:
                self.console.print(f"No more unique choices available for candidate {q_i+1}", style="red")
                break

            if progress:
                progress.log(f"Choosing candidate {q_i+1} of {q}", style="yellow")
            # 准备当前批次的数据 (Num_Valid, 1, D) - q-batch 维度设为 1 用于当前评估
            choices_batched = choices[valid_indices].unsqueeze(-2)
            # 计算采集函数值
            with torch.no_grad():
                # 使用 cholesky_jitter 增加数值稳定性
                with gpytorch.settings.cholesky_jitter(1e-3):
                    acq_values = self._split_batch_eval_acqf(
                        acq_func=acq_func,
                        X=choices_batched,
                        max_batch_size=max_batch_size,
                    )
            # 选择最佳点
            best_idx_in_batch = torch.argmax(acq_values)
            best_acq_val = acq_values[best_idx_in_batch]
            # 映射回全局索引
            best_global_idx = valid_indices[best_idx_in_batch]

            # (1, 1, D)
            selected_candidate = choices[best_global_idx].unsqueeze(0).unsqueeze(0)

            candidate_list.append(selected_candidate)
            acq_value_list.append(best_acq_val)

            # 更新 Pending Points
            # 形状变为 (1, current_q, D)
            current_candidates = torch.cat(candidate_list, dim=-2)
            torch.cuda.empty_cache()

            # 关键步骤：告诉采集函数这些点已经被选了，寻找下一个点时要基于这些点的条件分布
            acq_func.set_X_pending(current_candidates)
            # 如果要求唯一，更新掩码排除掉刚刚选中的点
            if unique:
                # 计算剩余点到最新选中点的距离
                dist_to_new = torch.norm(choices - selected_candidate.squeeze(), dim=-1)
                is_far_enough = dist_to_new > min_distance
                current_mask = current_mask & is_far_enough
            if progress:
                progress.update(task, advance=1)

            # 可选：打印调试信息
            # print(f"Batch {q_i}: Best Acq Value = {best_acq_val.item():.6e}")
        # 4. 清理与返回
        acq_func.set_X_pending(None)  # 防止副作用

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
        maximum_metrics: bool = True,  # EHVI 可能会用到这个参数来决定方向，但在 discrete 逻辑里不影响循环结构
        progress: object = None,
        task: object = None,
        min_distance: float = 1e-6,
        exclude_points: Tensor = None,
    ) -> tuple[Tensor, Tensor]:

        # 直接调用基类的通用逻辑
        # 注意：这里需要传入全局的 console 对象，假设你的环境中有一个名为 console 的变量
        return self.optimize_discrete_greedy(
            acq_func=self.acquisition_function,
            q=q,
            choices=choices,
            max_batch_size=max_batch_size,
            unique=unique,
            progress=progress,
            task=task,
            min_distance=min_distance,
            exclude_points=exclude_points,
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
    ) -> tuple[Tensor, Tensor]:
        # 直接调用基类的通用逻辑
        return self.optimize_discrete_greedy(
            acq_func=self.acquisition_function,
            q=q,
            choices=choices,
            max_batch_size=max_batch_size,
            unique=unique,
            progress=progress,
            task=task,
            min_distance=min_distance,
            exclude_points=exclude_points,
        )


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
