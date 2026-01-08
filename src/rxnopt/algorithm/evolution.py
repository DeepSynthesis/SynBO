from typing import List, Tuple, Dict
import numpy as np
import torch
from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn, TimeRemainingColumn

from botorch.models import ModelListGP
from botorch.utils.multi_objective.pareto import is_non_dominated

from rxnopt.utils.logger import console
from rxnopt.algorithm.sg_model import GPSurrogateModel

import warnings
from linear_operator.utils.cholesky import NumericalWarning

warnings.filterwarnings("ignore", category=NumericalWarning)


class DefaultEO:
    def __init__(
        self,
        random_seed: int = 42,
        surrogate_model: str = "GP",
        device: torch.device = torch.device("cpu"),
        accuracy: str = "medium",
    ):
        """
        初始化进化优化算法 (Evolutionary Optimization).
        这里采用基于代理模型的筛选策略 (Surrogate-Assisted Evolutionary Strategy).
        """
        self.random_seed = random_seed
        self.console = console
        self.device = device

        self.accuracy = accuracy

        if accuracy == "medium":
            self.pred_batch_size = 1024
        elif accuracy == "high":
            self.pred_batch_size = 512
        else:
            self.pred_batch_size = 2048

        if surrogate_model == "GP":
            self.surrogate_model_class = GPSurrogateModel
        else:
            raise ValueError(f"Unknown surrogate model: {surrogate_model}")

    def optimize(
        self,
        training_X: np.ndarray,
        training_y: np.ndarray,
        candidate_X: np.ndarray,
        opt_metric_settings: List[dict],
        batch_size: int,
        training_y_dict: dict,
    ) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:

        # 1. 数据预处理与张量转换
        training_X_t = torch.tensor(training_X).double().to(device=self.device)
        training_y_t = torch.tensor(training_y).double()
        # 对训练目标值进行加权处理
        training_y_t = self._weight_y(training_y_t, opt_metric_settings).to(device=self.device)
        candidate_X_t = torch.tensor(candidate_X).double().to(device=self.device)

        with Progress(
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:

            # 2. 训练代理模型 (与BO一致)
            num_models = training_y_t.shape[1]
            task_train = progress.add_task(description="Training surrogate models", total=num_models, start=True)

            models = []
            for i in range(num_models):
                key = list(training_y_dict.keys())[i]
                progress.log(f"Fitting model for [bold]{key}[/bold]...", style="yellow")

                train_y_i = training_y_t[:, i].reshape(-1, 1)
                model_i = self.surrogate_model_class(device=self.device, num_dims=training_X_t.shape[1])
                model_i.fit(training_X_t, train_y_i)
                models.append(model_i.model)
                progress.update(task_train, advance=1)

            self.global_model = ModelListGP(*models)

            # 3. 对所有候选点进行预测 (Model Inference)
            # EO的核心：用模型预测候选集的均值，作为"适应度(Fitness)"
            progress.log("Predicting mean values for all candidates...", style="blue")

            # 排除已有的训练点
            # 注意：在浮点数比较中通常需要一定的容差，这里简化处理，假设candidate包含未测点
            # 实际操作中，建议在传入 candidate_X 前就通过 Pandas 等工具去除已测点

            task_pred = progress.add_task(description="Predicting candidates", total=len(candidate_X_t))
            pred_means_list = []
            pred_vars_list = []

            # 分批次预测以节省显存
            with torch.no_grad():
                for i in range(0, len(candidate_X_t), self.pred_batch_size):
                    batch_X = candidate_X_t[i : i + self.pred_batch_size]
                    posterior = self.global_model.posterior(batch_X)
                    pred_means_list.append(posterior.mean)
                    pred_vars_list.append(posterior.variance)
                    progress.update(task_pred, advance=len(batch_X))

            all_pred_mean = torch.cat(pred_means_list, dim=0)  # Shape: (N_cand, N_obj)
            all_pred_var = torch.cat(pred_vars_list, dim=0)
            all_pred_std = torch.sqrt(all_pred_var)

            # 4. NSGA-II 排序选择 (Sorting and Selection)
            # 为了方便排序，我们需要根据 opt_metric_settings 将所有目标统一为"最大化"问题
            # 已经在 _weight_y 中处理了权重，但 min/max 方向需要在排序时处理

            progress.log("Performing NSGA-II selection...", style="magenta")

            # 构建用于排序的 fitness tensor (统一转为 Maximize)
            fitness = all_pred_mean.clone()
            for i, setting in enumerate(opt_metric_settings):
                if setting["opt_direct"] == "min":
                    fitness[:, i] = -fitness[:, i]

            # 执行非支配排序筛选
            selected_indices = self._nsga2_selection(fitness, batch_size)

            # 提取结果
            selected_indices_cpu = selected_indices.cpu().numpy()
            best_samples = candidate_X[selected_indices_cpu]

            # 获取选中点的预测结果
            best_pred_mean_t = all_pred_mean[selected_indices]
            best_pred_std_t = all_pred_std[selected_indices]

        # 5. 结果后处理
        # 还原 min/max 的符号 (仅用于展示)
        # 注意：这里返回的 pred_mean 应该是原始物理意义的值，所以如果之前取反了要变回来，
        # 并且要除以权重

        # EO 倾向于利用(Exploit)模型预测最好的点，这里统一标记为 pareto_best
        recommend_type = ["pareto_best"] * batch_size

        # 反标准化/反权重处理
        pred_mean_np = self._unweight_y(best_pred_mean_t, opt_metric_settings).cpu().numpy()
        pred_std_np = self._unweight_y(best_pred_std_t, opt_metric_settings).cpu().numpy()

        # 对于 min 目标，预测值在 unweight 之后已经是原始尺度了 (因为 _weight_y 没有处理符号)，
        # 所以不需要像 optimize 内部那样取反。
        # 但是，如果在 opt_metric_settings 中有特殊处理逻辑，需保持一致。
        # 遵循 DefaultBO 的逻辑：DefaultBO 在最后输出前做了一个 check:
        # if d["opt_direct"] == "min": pred_mean[:, i] = -pred_mean[:, i] (如果内部为了优化取反过)
        # 在本代码中，_weight_y 没有取反，只在 NSGA2 排序前的 fitness 变量里取反了。
        # 所以 all_pred_mean 还是正向的（仅加权）。
        # 因此，直接返回 unweight 后的值即可，无需符号反转。

        return best_samples, recommend_type, pred_mean_np, pred_std_np

    def _nsga2_selection(self, fitness: torch.Tensor, n_select: int) -> torch.Tensor:
        """
        实现简化的 NSGA-II 选择逻辑：
        1. 非支配排序分层 (Non-dominated Sorting)
        2. 拥挤距离计算 (Crowding Distance)

        Args:
            fitness: (N, M) tensor, 假设所有目标都是求最大化 (Maximize).
            n_select: 需要选择的样本数量.

        Returns:
            selected_indices: (n_select,) tensor, 选中样本在原数据中的索引.
        """
        n_points = fitness.shape[0]
        indices = torch.arange(n_points, device=fitness.device)
        selected_indices = []

        # 1. 快速非支配排序
        # 使用 mask 逐步剥离 Pareto Front
        current_indices = indices
        current_fitness = fitness

        while len(selected_indices) < n_select and len(current_indices) > 0:
            # botorch 的 is_non_dominated 默认假设最大化
            # dedup=False 表示不合并重复点，保持索引对应
            pareto_mask = is_non_dominated(current_fitness)

            front_indices = current_indices[pareto_mask]

            # 如果当前层加入后不超过需求，全选
            if len(selected_indices) + len(front_indices) <= n_select:
                selected_indices.extend(front_indices.tolist())
                # 从剩余池中移除
                remaining_mask = ~pareto_mask
                current_indices = current_indices[remaining_mask]
                current_fitness = current_fitness[remaining_mask]
            else:
                # 2. 拥挤距离排序 (最后这一层选不完了，需要挑稀疏的)
                n_needed = n_select - len(selected_indices)

                # 计算当前 Front 的拥挤距离
                # 提取当前层的 fitness
                front_fitness = fitness[front_indices]  # (N_front, M)

                # 计算距离
                crowding_dists = self._calc_crowding_distance(front_fitness)

                # 降序排列 (距离越大越稀疏，越好)
                sorted_vals, sorted_idxs = torch.sort(crowding_dists, descending=True)

                # 取前 n_needed 个
                best_in_front_indices = front_indices[sorted_idxs[:n_needed]]
                selected_indices.extend(best_in_front_indices.tolist())
                break

        return torch.tensor(selected_indices, device=fitness.device, dtype=torch.long)

    def _calc_crowding_distance(self, front_fitness: torch.Tensor) -> torch.Tensor:
        """
        计算拥挤距离
        """
        n_points, n_objs = front_fitness.shape
        if n_points == 0:
            return torch.tensor([], device=front_fitness.device)

        # 初始化距离为 0
        distances = torch.zeros(n_points, device=front_fitness.device)

        for i in range(n_objs):
            # 对第 i 个目标值进行排序
            obj_values, sorted_indices = torch.sort(front_fitness[:, i])

            # 边界点的距离设为无穷大 (保持边界)
            distances[sorted_indices[0]] = float("inf")
            distances[sorted_indices[-1]] = float("inf")

            # 归一化因子 (Max - Min)
            norm = obj_values[-1] - obj_values[0]
            if norm == 0:
                norm = 1e-6

            # 中间点的距离累加: (f[k+1] - f[k-1]) / (f_max - f_min)
            # 这里的 obj_values 已经是排序过的
            if n_points > 2:
                diffs = (obj_values[2:] - obj_values[:-2]) / norm
                # 需要加回原来的索引位置
                # sorted_indices[1:-1] 是中间点的原始索引
                distances[sorted_indices[1:-1]] += diffs

        return distances

    def _weight_y(self, training_y: torch.Tensor, opt_metric_settings: List[dict]) -> torch.Tensor:
        """与 DefaultBO 保持一致的加权逻辑"""
        weights = torch.tensor([d["metric_weight"] for d in opt_metric_settings], device=training_y.device)
        training_y = training_y * weights
        return training_y

    def _unweight_y(self, training_y: torch.Tensor, opt_metric_settings: List[dict]) -> torch.Tensor:
        """与 DefaultBO 保持一致的反加权逻辑"""
        weights = torch.tensor([d["metric_weight"] for d in opt_metric_settings], device=training_y.device)
        training_y = training_y / weights
        return training_y
