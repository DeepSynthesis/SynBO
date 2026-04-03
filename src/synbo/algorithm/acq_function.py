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
        unused_reagent_boost: Tensor = None,
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
            unused_reagent_boost: Tensor of shape (n,) containing boost values for candidates with unused reagents.
                                 Higher values give more priority to candidates with unused reagents.

        Returns:
            tuple[Tensor, Tensor]:
                - Selected candidates (q, D)
                - Acquisition values for selected candidates (q,)
        """

        # acq_result = optimize_acqf_discrete(
        #     acq_function=acq_func,
        #     choices=choices,
        #     q=q,
        #     unique=unique,
        #     max_batch_size=max_batch_size,
        # )

        acq_result = self._new_optimize_acqf_discrete(
            acq_function=acq_func,
            choices=choices,
            q=q,
            unique=unique,
            max_batch_size=max_batch_size,
            unused_reagent_boost=unused_reagent_boost,
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
        unused_reagent_boost: Tensor = None,
        progress: object = None,
        task: object = None,
    ):

        def _split_batch_eval_acqf(acq_function, X, max_batch_size):
            acq_values_list = []
            for X_batch in X.split(max_batch_size):
                X_batch_gpu = X_batch.to(device=self.device)  # 仅小批量上 GPU
                acq_batch = acq_function(X_batch_gpu)
                acq_values_list.append(acq_batch.cpu())  # 立即移回 CPU
                del X_batch_gpu  # 显式释放
            return torch.cat(acq_values_list, dim=0)  # 在 CPU 上连接

        choices_batched = choices.unsqueeze(-2)
        if q > 1:
            candidate_list, acq_value_list = [], []
            base_X_pending = acq_function.X_pending
            for q_i in range(q):
                if progress and task:
                    progress.update(task, advance=1)
                with torch.no_grad():
                    acq_values = _split_batch_eval_acqf(
                        acq_function=acq_function,
                        X=choices_batched,
                        max_batch_size=max_batch_size,
                    )

                # Apply unused reagent boost if provided
                if unused_reagent_boost is not None:
                    # Get the current valid indices (choices_batched may have been reduced)
                    num_choices = len(choices_batched)
                    # Map to original indices based on how many have been removed

                    # Get boost values for current choices
                    # current_boost = unused_reagent_boost[start_idx:end_idx].to(acq_values.device)
                    # Add boost to acquisition values

                    acq_values = acq_values + unused_reagent_boost.to(acq_values.device)

                # === 新增：应用多样性boost ===
                diversity_weight = 1
                if q_i > 0:
                    diversity_boost = self._compute_diversity_boost(choices_batched.squeeze(1), candidate_list, acq_values.device)
                    acq_values = acq_values + diversity_weight * diversity_boost
                # =============================

                best_idx = torch.argmax(acq_values)

                candidate_list.append(choices_batched[best_idx])
                acq_value_list.append(acq_values[best_idx])
                # set pending points
                candidates = torch.cat(candidate_list, dim=-2)

                # need to remove choice from choice set if enforcing uniqueness
                if unique:
                    choices_batched = torch.cat([choices_batched[:best_idx], choices_batched[best_idx + 1 :]])
                    # Also update unused_reagent_boost to match
                    if unused_reagent_boost is not None:
                        # Remove the selected index from boost tensor
                        unused_reagent_boost = torch.cat([unused_reagent_boost[:best_idx], unused_reagent_boost[best_idx + 1 :]])

            # Reset acq_func to previous X_pending state
            acq_function.set_X_pending(base_X_pending)
            return candidates, torch.stack(acq_value_list)

        with torch.no_grad():
            acq_values = _split_batch_eval_acqf(acq_function=acq_function, X=choices_batched, max_batch_size=max_batch_size)

        # Apply unused reagent boost if provided
        if unused_reagent_boost is not None:
            acq_values = acq_values + unused_reagent_boost.to(acq_values.device)

        best_idx = torch.argmax(acq_values)
        return choices_batched[best_idx], acq_values[best_idx]

    def _compute_diversity_boost(
        self,
        choices: torch.Tensor,
        selected_candidates: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """计算多样性奖励：与已选点距离越远，奖励越高"""
        if not selected_candidates:
            return torch.zeros(len(choices), device=device)

        selected = torch.cat(selected_candidates, dim=-2)# .squeeze(0)
        distances = torch.cdist(choices, selected)
        min_distances = distances.min(dim=1)[0]  # 到最近已选点的距离

        # 距离越大，boost越高（归一化后）
        boost = (min_distances - min_distances.min()) / (min_distances.max() - min_distances.min() + 1e-8)
        return boost


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
        self.acquisition_function = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            sampler=sampler,
            ref_point=ref_point,
            # partitioning=partitioning,
            alpha=0.0,
            incremental_nehvi=True,
            X_baseline=train_x,
            prune_baseline=True,
        )

        print(self.ref_point)

    def optimize_acqf_discrete(
        self,
        q: int,
        choices: Tensor,
        max_batch_size: int = 1024,
        unique: bool = True,
        progress: object = None,
        task: object = None,
        unused_reagent_boost: Tensor = None,
    ) -> tuple[Tensor, Tensor]:

        return self.optimize_discrete(
            acq_func=self.acquisition_function,
            q=q,
            choices=choices,
            max_batch_size=max_batch_size,
            unique=unique,
            progress=progress,
            task=task,
            unused_reagent_boost=unused_reagent_boost,
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
        maximum_metrics: bool = True,
        progress: object = None,
        task: object = None,
        unused_reagent_boost: Tensor = None,
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
            unused_reagent_boost=unused_reagent_boost,
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
        unused_reagent_boost: Tensor = None,
    ) -> tuple[Tensor, Tensor]:
        return self.optimize_discrete(
            acq_func=self.acquisition_function,
            q=q,
            choices=choices,
            max_batch_size=max_batch_size,
            unique=unique,
            progress=progress,
            task=task,
            unused_reagent_boost=unused_reagent_boost,
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
        device,
        weights: Tensor = None,
        prune_baseline: bool = True,  # 这是一个通常有用的选项，用于减少计算量
    ):
        super().__init__(model, sampler, device)

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
        max_batch_size: int = 1024,
        unique: bool = True,
        maximum_metrics: bool = True,
        progress: object = None,
        task: object = None,
        unused_reagent_boost: Tensor = None,
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
            unused_reagent_boost=unused_reagent_boost,
        )


from botorch.acquisition.multi_objective.max_value_entropy_search import (
    qLowerBoundMultiObjectiveMaxValueEntropySearch,
)
from botorch.acquisition.multi_objective.utils import compute_sample_box_decomposition
from botorch.utils.multi_objective.pareto import is_non_dominated


class MOGIBBONAcquisitionFunction(BaseAcquisitionFunction):
    """
    Multi-Objective Lower Bound Max-Value Entropy Search (MO-GIBBON).
    与 EHVIAcquisitionFunction 的关键区别:
    ----------------------------------------------------------
    | 项目                   | EHVI          | 本类                              |
    |------------------------|---------------|-----------------------------------|
    | hypercell_bounds shape | 2 × J × M     | num_pareto_samples × 2 × J × M    |
    | 来源                   | 已观测 Pareto  | GP posterior 多次采样的 Pareto 前沿 |
    | 体现 Pareto 不确定性    | 否             | 是                                |
    ----------------------------------------------------------
    关于 candidate_set_for_sampling 参数:
        用于从 GP posterior 采样 Pareto 前沿的候选集。
        - 默认 None → 自动使用训练数据（几十个点，计算快但覆盖有限）
        - 建议: 传入 choices 的随机子集（如 512~2048 个点），
          可更好地覆盖候选空间，使 Pareto 前沿估计更准确。
    关于 estimation_type:
        "LB"  (默认): 解析高斯下界，速度快，推荐。
        "LB2"       : 更紧的下界，稍慢。
        "MC"        : Monte Carlo 估计，最慢但最准。
        "0"         : 最简单的下界。
    Notes:
        (i)  采集值可能为负数 —— 这是数学性质，不是 bug。
        (ii) batch q>1 时，增加更多候选点不保证单调增加采集值。
    """

    def __init__(
        self,
        model: ModelListGP,
        sampler: SobolQMCNormalSampler,
        ref_point: torch.Tensor,
        partitioning,  # 仅保留以兼容调用接口，不在内部使用
        num_pareto_samples: int = 10,  # 采样 Pareto 前沿的数量
        num_pareto_points: int = 10,  # 每条 Pareto 前沿保留的点数
        estimation_type: str = "LB",
        num_samples: int = 64,
        candidate_set_for_sampling: Tensor = None,  # 用于采样 Pareto 前沿的候选集 (n, D)
    ):
        super().__init__(model, sampler)
        self.ref_point = ref_point
        self.partitioning = partitioning
        # ── 确定用于 Pareto 前沿采样的候选集 ────────────────────────────────
        # 优先使用外部传入的 candidate_set_for_sampling
        # 若未提供，退回到训练数据（适合先验数据少的早期阶段）
        if candidate_set_for_sampling is not None:
            candidate_set = candidate_set_for_sampling
        else:
            # ModelListGP: 所有子模型共享同一 train_inputs
            candidate_set = model.models[0].train_inputs[0]  # (n_train, D)
        # ── Step 1: 从 GP posterior 采样多条 Pareto 前沿 ───────────────────
        # pareto_fronts shape: (num_pareto_samples, num_pareto_points, M)
        pareto_fronts = self._sample_pareto_frontiers(
            model=model,
            candidate_set=candidate_set,
            num_pareto_samples=num_pareto_samples,
            num_pareto_points=num_pareto_points,
            ref_point=ref_point,
        )
        # ── Step 2: 对每条 Pareto 前沿做 box decomposition ─────────────────
        # hypercell_bounds shape: (num_pareto_samples, 2, J, M)
        # 这就是 qLowerBoundMultiObjectiveMaxValueEntropySearch 所需的格式
        hypercell_bounds = compute_sample_box_decomposition(
            pareto_fronts=pareto_fronts,
        )
        # ── Step 3: 构建采集函数 ────────────────────────────────────────────
        self.acquisition_function = qLowerBoundMultiObjectiveMaxValueEntropySearch(
            model=model,
            hypercell_bounds=hypercell_bounds,
            estimation_type=estimation_type,
            num_samples=num_samples,
        )
        # ── set_X_pending shim ───────────────────────────────────────────────
        # 与 MCAcquisitionFunction 不同，此类可能未暴露 set_X_pending()
        # 注入 shim 使 optimize_discrete / optimize_discrete_unuse 中的
        # acq_func.set_X_pending(...) 调用不会抛出 AttributeError
        if not hasattr(self.acquisition_function, "set_X_pending"):
            _acq = self.acquisition_function

            def _set_x_pending(X_pending: torch.Tensor | None = None) -> None:
                # @concatenate_pending_points 装饰器直接读取 self.X_pending 属性
                _acq.X_pending = X_pending.clone().detach() if X_pending is not None else None

            _acq.set_X_pending = _set_x_pending
        print(self.ref_point)

    @staticmethod
    def _sample_pareto_frontiers(
        model: ModelListGP,
        candidate_set: Tensor,
        num_pareto_samples: int,
        num_pareto_points: int,
        ref_point: Tensor,
    ) -> Tensor:
        """
        在 candidate_set 上对 GP posterior 进行采样，
        从每次采样中提取 Pareto 最优点，最终返回固定大小的 Pareto 前沿张量。
        Args:
            model:              已拟合的 ModelListGP。
            candidate_set:      形状 (n, D)，从中采样的候选点集。
            num_pareto_samples: 要采样的 Pareto 前沿数量。
            num_pareto_points:  每条 Pareto 前沿固定保留的点数。
                                超出时随机下采样，不足时有放回重复采样填充。
            ref_point:          形状 (M,)，参考点；严格低于此点的样本会被过滤掉。
        Returns:
            形状 (num_pareto_samples, num_pareto_points, M) 的 Tensor。
        """
        device = candidate_set.device
        dtype = candidate_set.dtype
        ref = ref_point.to(device=device, dtype=dtype)
        with torch.no_grad():
            posterior = model.posterior(candidate_set)
            # samples: (num_pareto_samples, n_candidates, M)
            samples = posterior.rsample(torch.Size([num_pareto_samples]))
        M = samples.shape[-1]
        pareto_fronts = torch.zeros(
            num_pareto_samples,
            num_pareto_points,
            M,
            device=device,
            dtype=dtype,
        )
        for i in range(num_pareto_samples):
            y_i = samples[i]  # (n_candidates, M)
            # 过滤掉在 ref_point 以下的点（这些点对 hypervolume 无贡献）
            above_ref_mask = (y_i > ref.unsqueeze(0)).all(dim=-1)
            y_feasible = y_i[above_ref_mask]
            if y_feasible.shape[0] == 0:
                # 边界情况: 无点超过 ref_point
                # 退化策略: 取所有目标之和排名最高的 num_pareto_points 个点
                scores = y_i.sum(dim=-1)
                k = min(num_pareto_points, y_i.shape[0])
                top_idx = scores.topk(k).indices
                pareto_fronts[i, :k] = y_i[top_idx]
                if k < num_pareto_points:
                    # 用最后一个点重复填充（极罕见情况）
                    pareto_fronts[i, k:] = pareto_fronts[i, k - 1].unsqueeze(0).expand(num_pareto_points - k, M)
                continue
            # 找出非支配点 (Pareto 最优)
            is_nd = is_non_dominated(y_feasible)  # (n_feasible,) bool
            pareto_pts = y_feasible[is_nd]  # (n_pareto, M)
            n_pareto = pareto_pts.shape[0]
            if n_pareto >= num_pareto_points:
                # 随机下采样到 num_pareto_points（无放回）
                idx = torch.randperm(n_pareto, device=device)[:num_pareto_points]
                pareto_fronts[i] = pareto_pts[idx]
            else:
                # 先放入所有 Pareto 点，再有放回重复采样填充剩余位置
                pareto_fronts[i, :n_pareto] = pareto_pts
                if n_pareto > 0:
                    fill_idx = torch.randint(
                        n_pareto,
                        (num_pareto_points - n_pareto,),
                        device=device,
                    )
                    pareto_fronts[i, n_pareto:] = pareto_pts[fill_idx]
        return pareto_fronts  # (num_pareto_samples, num_pareto_points, M)

    def optimize_acqf_discrete(
        self,
        q: int,
        choices: Tensor,
        max_batch_size: int = 1024,
        unique: bool = True,
        progress: object = None,
        task: object = None,
        unused_reagent_boost: Tensor = None,
    ) -> tuple[Tensor, Tensor]:
        return self.optimize_discrete(
            acq_func=self.acquisition_function,
            q=q,
            choices=choices,
            max_batch_size=max_batch_size,
            unique=unique,
            progress=progress,
            task=task,
            unused_reagent_boost=unused_reagent_boost,
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
