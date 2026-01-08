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

        # ================= [新增/修改部分 Start] =================
        # 0. 数据去重 (Deduplication)
        # 目的：确保候选集 candidate_X 中不包含 training_X 已经测过的点
        
        # 将 training_X 的每一行转为 tuple 并存入 set，利用哈希实现 O(1) 查找
        # 注意：这里使用的是精确匹配。如果数据是浮点数且存在微小误差，
        # 可能需要使用 np.isclose 或 torch.cdist 设置阈值过滤，但在大多数组合优化场景下，
        # 精确匹配（Set-based）是标准做法。
        existing_set = set(map(tuple, training_X))

        # 创建布尔掩码：如果 candidate_X 的某一行不在 existing_set 中，则保留
        # 使用列表推导式转换 tuple 比 np.apply_along_axis 在此场景下通常更快
        keep_mask = np.array([tuple(x) not in existing_set for x in candidate_X])

        # 过滤 candidate_X
        n_original = len(candidate_X)
        candidate_X = candidate_X[keep_mask]
        n_filtered = len(candidate_X)

        if n_original != n_filtered:
            self.console.log(f"Removed {n_original - n_filtered} duplicates from candidate set.", style="dim")

        # 边界情况处理：如果过滤后候选集为空
        if n_filtered == 0:
            warnings.warn("All candidates appear in training data. Returning empty results.")
            return np.array([]), [], np.array([]), np.array([])
            
        # 边界情况处理：如果过滤后样本数少于 batch_size，调整 batch_size
        current_batch_size = min(batch_size, n_filtered)
        # ================= [新增/修改部分 End] =================

        # 1. 数据预处理与张量转换
        training_X_t = torch.tensor(training_X).double().to(device=self.device)
        training_y_t = torch.tensor(training_y).double()
        # 对训练目标值进行加权处理
        training_y_t = self._weight_y(training_y_t, opt_metric_settings).to(device=self.device)
        
        # 注意：这里的 candidate_X 已经是去重后的 numpy 数组
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
            progress.log("Predicting mean values for all candidates...", style="blue")

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
            progress.log("Performing NSGA-II selection...", style="magenta")

            # 构建用于排序的 fitness tensor (统一转为 Maximize)
            fitness = all_pred_mean.clone()
            for i, setting in enumerate(opt_metric_settings):
                if setting["opt_direct"] == "min":
                    fitness[:, i] = -fitness[:, i]

            # 执行非支配排序筛选
            # 使用修正后的 current_batch_size 防止索引越界
            selected_indices = self._nsga2_selection(fitness, current_batch_size)

            # 提取结果
            selected_indices_cpu = selected_indices.cpu().numpy()
            
            # 这里的 candidate_X 是已经去重后的，所以提取出的 best_samples 也是干净的
            best_samples = candidate_X[selected_indices_cpu]

            # 获取选中点的预测结果
            best_pred_mean_t = all_pred_mean[selected_indices]
            best_pred_std_t = all_pred_std[selected_indices]

        # 5. 结果后处理
        recommend_type = ["pareto_best"] * len(best_samples)

        # 反标准化/反权重处理
        pred_mean_np = self._unweight_y(best_pred_mean_t, opt_metric_settings).cpu().numpy()
        pred_std_np = self._unweight_y(best_pred_std_t, opt_metric_settings).cpu().numpy()

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
        # 如果样本数本身就少于需求数，直接返回所有索引
        if n_points <= n_select:
            return torch.arange(n_points, device=fitness.device, dtype=torch.long)
            
        indices = torch.arange(n_points, device=fitness.device)
        selected_indices = []

        # 1. 快速非支配排序
        # 使用 mask 逐步剥离 Pareto Front
        current_indices = indices
        current_fitness = fitness

        while len(selected_indices) < n_select and len(current_indices) > 0:
            # botorch 的 is_non_dominated 默认假设最大化
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
