from typing import List, Tuple, Dict
import numpy as np
import torch
from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn, TimeRemainingColumn

from botorch.models import ModelListGP
from botorch.utils.multi_objective.pareto import is_non_dominated

from rxnopt.utils.logger import console
from rxnopt.algorithm.sg_model import (
    BNNEnsembleSurrogateModel,
    BayesianLinearSurrogateModel,
    GPSurrogateModel,
    RFSurrogateModel,
    SklearnModelWrapper,
)

import warnings
from linear_operator.utils.cholesky import NumericalWarning

warnings.filterwarnings("ignore", category=NumericalWarning)


class DefaultEO:
    def __init__(
        self,
        random_seed: int = 42,
        surrogate_model: str = "GP",
        device: torch.device = torch.device("cpu"),
        method: str = "Standard",  # Options: "Standard" (or "GA"), "Thompson"
        accuracy: str = "medium",
    ):
        """
        初始化进化优化算法 (Evolutionary Optimization).
        这里采用基于代理模型的筛选策略 (Surrogate-Assisted Evolutionary Strategy).

        Args:
            method:
                - "Standard" (或 "GA"): 使用代理模型的均值进行确定性筛选 (Exploitation)。
                - "Thompson": 使用 Thompson Sampling 从后验分布采样进行概率性筛选 (Exploration/Mutation)。
        """
        self.random_seed = random_seed
        self.console = console
        self.device = device
        self.method = method  # Store the method

        self.accuracy = accuracy

        if accuracy == "medium":
            self.pred_batch_size = 1024
        elif accuracy == "high":
            self.pred_batch_size = 512
        else:
            self.pred_batch_size = 2048

        if surrogate_model == "GP":
            self.surrogate_model_class = GPSurrogateModel
        elif surrogate_model == "RF":
            self.surrogate_model_class = RFSurrogateModel
        elif surrogate_model == "ensemble":
            self.surrogate_model_class = BNNEnsembleSurrogateModel
        elif surrogate_model == "linear":
            self.surrogate_model_class = BayesianLinearSurrogateModel
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

        # 1. Data deduplication
        existing_set = set(map(tuple, training_X))
        keep_mask = np.array([tuple(x) not in existing_set for x in candidate_X])

        n_original = len(candidate_X)
        candidate_X = candidate_X[keep_mask]
        n_filtered = len(candidate_X)

        if n_original != n_filtered:
            self.console.log(f"Removed {n_original - n_filtered} duplicates from candidate set.", style="dim")

        if n_filtered == 0:
            warnings.warn("All candidates appear in training data. Returning empty results.")
            return np.array([]), [], np.array([]), np.array([])

        current_batch_size = min(batch_size, n_filtered)

        # 2. Tensor conversion & weighting
        training_X_t = torch.tensor(training_X).double().to(device=self.device)
        training_y_t = torch.tensor(training_y).double()
        training_y_t = self._weight_y(training_y_t, opt_metric_settings).to(device=self.device)
        candidate_X_t = torch.tensor(candidate_X).double().to(device=self.device)

        with Progress(
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:

            # 3. Train Surrogate Models
            num_models = training_y_t.shape[1]
            task_train = progress.add_task(description="Training surrogate models", total=num_models, start=True)

            models = []
            for i in range(num_models):
                key = list(training_y_dict.keys())[i]
                progress.log(f"Fitting model for [bold]{key}[/bold]...", style="yellow")

                train_y_i = training_y_t[:, i].reshape(-1, 1)
                model_i = self.surrogate_model_class(device=self.device, num_dims=training_X_t.shape[1])

                if isinstance(model_i, GPSurrogateModel):
                    model_i.fit(training_X_t, train_y_i)
                    models.append(model_i.model)
                else:
                    wrapper = SklearnModelWrapper(model_i)
                    wrapper.fit_surrogate(training_X_t, train_y_i)
                    models.append(wrapper)

                progress.update(task_train, advance=1)

            self.global_model = ModelListGP(*models)

            task_pred = progress.add_task(description="Predicting candidates", total=len(candidate_X_t))
            pred_means_list = []
            pred_vars_list = []

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

            for it, settings in enumerate(opt_metric_settings):
                low, high = (0, 1) if settings["opt_direct"] == "max" else (-1, 0)
                all_pred_mean[:, it].clamp_(min=low, max=high)

            # 5. Selection Strategy (Standard vs Thompson)
            progress.log(f"Performing NSGA-II selection ({self.method})...", style="magenta")

            if self.method == "Thompson":
                # Thompson Sampling: Sample from Normal(mean, std)
                # 相当于引入了有方向的随机变异，不确定性越大的点，波动的幅度越大
                fitness = torch.normal(mean=all_pred_mean, std=all_pred_std)
                rec_label = "Evolution (Thompson)"
            else:
                # Standard / GA: Use Mean directly (Greedy / Exploitation)
                fitness = all_pred_mean.clone()
                rec_label = "Evolution (Standard)"

            selected_indices = self._nsga2_selection(fitness, current_batch_size)
            selected_indices_cpu = selected_indices.cpu().numpy()

            best_samples = candidate_X[selected_indices_cpu]

            # 获取选中样本的 Mean/Std 用于展示（注意：即使是 Thompson，返回给用户的预测值通常还是 Mean）
            best_pred_mean_t = all_pred_mean[selected_indices]
            best_pred_std_t = all_pred_std[selected_indices]

        recommend_type = [rec_label] * len(best_samples)

        pred_mean_np = self._unweight_y(best_pred_mean_t, opt_metric_settings).cpu().numpy()
        pred_std_np = self._unweight_y(best_pred_std_t, opt_metric_settings).cpu().numpy()

        return best_samples, recommend_type, pred_mean_np, pred_std_np

    def _nsga2_selection(self, fitness: torch.Tensor, n_select: int) -> torch.Tensor:
        """
        实现简化的 NSGA-II 选择逻辑，并打印每一层帕累托前沿的信息。
        """
        print(111)
        n_points = fitness.shape[0]
        # 如果样本数本身就少于需求数，直接返回所有索引
        if n_points <= n_select:
            return torch.arange(n_points, device=fitness.device, dtype=torch.long)

        indices = torch.arange(n_points, device=fitness.device)
        selected_indices = []

        # 快速非支配排序
        # 使用 mask 逐步剥离 Pareto Front
        current_indices = indices
        current_fitness = fitness

        # 用于记录当前是第几层 Front
        front_rank = 0

        while len(selected_indices) < n_select and len(current_indices) > 0:
            pareto_mask = is_non_dominated(current_fitness)
            front_indices = current_indices[pareto_mask]

            # =========== 添加的打印逻辑 START ===========
            # 获取当前 Front 对应的具体 Fitness 值
            this_front_values = fitness[front_indices]

            print(f"\n[NSGA-II Selection] Pareto Front Rank: {front_rank}")
            print(f"  > Number of points: {len(front_indices)}")
            print(f"  > Indices in original population: {front_indices.tolist()}")
            # 打印具体的 Fitness 值 (建议保留一定的小数精度以便查看)
            # 这里直接打印 Tensor，PyTorch 会自动格式化
            print(f"  > Fitness Values:\n{this_front_values}")

            front_rank += 1
            # =========== 添加的打印逻辑 END =============

            if len(selected_indices) + len(front_indices) <= n_select:
                selected_indices.extend(front_indices.tolist())
                remaining_mask = ~pareto_mask
                current_indices = current_indices[remaining_mask]
                current_fitness = current_fitness[remaining_mask]
            else:
                n_needed = n_select - len(selected_indices)
                front_fitness = fitness[front_indices]
                crowding_dists = self._calc_crowding_distance(front_fitness)

                sorted_vals, sorted_idxs = torch.sort(crowding_dists, descending=True)

                # 这一步是为了打印最后被截断层中实际被选中的点（可选，如果需要知道最后谁被选中了）
                # print(f"  > (Crowding Distance Applied) Keeping top {n_needed} from this front.")

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

        distances = torch.zeros(n_points, device=front_fitness.device)

        for i in range(n_objs):
            obj_values, sorted_indices = torch.sort(front_fitness[:, i])

            distances[sorted_indices[0]] = float("inf")
            distances[sorted_indices[-1]] = float("inf")

            norm = obj_values[-1] - obj_values[0]
            if norm == 0:
                norm = 1e-6

            if n_points > 2:
                diffs = (obj_values[2:] - obj_values[:-2]) / norm
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
