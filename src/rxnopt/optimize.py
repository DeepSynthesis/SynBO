from typing import List
import numpy as np
import torch
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn, TimeRemainingColumn


from botorch.models import ModelListGP
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.sampling.normal import SobolQMCNormalSampler

from .utils.util_func import compute_hvi

from .bo_algorithm.GP_opt import GPSurrogateModel
from .bo_algorithm.acf_opt import optimize_acqf_discrete, EHVIAcquisitionFunction, ParetoFrontCalculator

# 在文件顶部导入警告模块和具体警告类
import warnings
from linear_operator.utils.cholesky import NumericalWarning

# 静音该警告（全局生效，或按需用上下文管理器）
warnings.filterwarnings("ignore", category=NumericalWarning)


class Optimizer:
    def __init__(self, method, name_data, mc_num_samples: int = 64, max_batch_size: int = 128, seed: int = 1145141):
        """_summary_

        Args:
            method (_type_): _description_
            name_data (_type_): _description_
            mc_num_samples (int, optional): . Defaults to 64.
            max_batch_size (int, optional): _description_. Defaults to 128.
            seed (int, optional): _description_. Defaults to 1145141.
        """
        self.method = method
        self.name_data = name_data
        self.mc_num_samples = mc_num_samples
        self.seed = seed
        self.surrogate_model_class = GPSurrogateModel
        self.acquisition_function_class = EHVIAcquisitionFunction
        self.target_evaluator = ParetoFrontCalculator()
        self.max_batch_size = max_batch_size

        self.opt_console = Console()

    def optimize(
        self,
        training_X: np.ndarray,
        training_y: np.ndarray,
        candidate_X: np.ndarray,
        opt_direct_info: List[dict],
        device: torch.device,
        batch_size: int = 5,
        opt_weights: dict = None,
        maximum_metrics: bool = True,
    ) -> List[int]:
        """
        Core Bayesian Optimization routine

        Args:
            train_x: Normalized training inputs (numpy array)
            train_y: Normalized training outputs (numpy array)
            candidate_x: All possible candidate points (numpy array)
            batch_size: Number of points to select

        Returns:
            Indices of selected points from candidate_x
        """
        # Convert to tensors
        # TODO: deal with weights

        # solved the maximum problem by transforming the training_y to a minimum problem
        for k, d in zip(training_y.keys(), opt_direct_info):
            if d["opt_direct"] == "min":
                training_y[k] = -training_y[k]

        training_y_dict = training_y.copy()
        if isinstance(training_y, dict):
            training_y = np.array(list(training_y.values())).T
        training_X_t = torch.tensor(training_X).double()
        training_y_t = torch.tensor(training_y).double()
        candidate_X_t = torch.tensor(candidate_X).double()
        dtype = torch.double
        training_X_t = training_X_t.to(device=device, dtype=dtype)
        training_y_t = training_y_t.to(device=device, dtype=dtype)
        candidate_X_t = candidate_X_t.to(device=device, dtype=dtype)
        models = []

        with Progress(
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=self.opt_console,
        ) as progress:
            # training surrogate models
            num_models = training_y_t.shape[1]
            task_train = progress.add_task(description="Training surrogate models", total=num_models, start=True)
            for i in range(num_models):
                key = list(training_y_dict.keys())[i]
                progress.log(f"Fitting model for [bold]{key}[/bold]...", style="yellow")

                train_y_i = training_y_t[:, i].reshape(-1, 1)
                model_i = self.surrogate_model_class(device=device, num_dims=training_X.shape[1])
                model_i.fit(training_X_t, train_y_i)
                models.append(model_i.model)
                progress.update(task_train, advance=1)
            self.global_model = ModelListGP(*models)

            # calculate pareto frontiers
            task_pareto = progress.add_task(description="Calculating Pareto frontiers", total=len(training_y) - 1)
            self.pareto_y = self.target_evaluator.calculate_target_function(training_y, progress, task_pareto).to(device=device)

            # ref_point is in the upper right corner.
            # That is, if opt_metric_info in opt_direct is max, take 0; if it is min, take 1
            self.ref_point = torch.tensor([-1 if omi["opt_direct"] == "min" else 0 for omi in opt_direct_info], dtype=float, device=device)

            # Adaptive MC sampling based on dimensionality and problem size
            # TODO: remove this adaptive MC sample strategy
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.mc_num_samples]), seed=self.seed)
            partitioning = NondominatedPartitioning(ref_point=self.ref_point, Y=self.pareto_y)
            acq_func = self.acquisition_function_class(
                model=self.global_model,
                sampler=sampler,
                ref_point=self.ref_point,
                partitioning=partitioning,
                maximum_metrics=maximum_metrics,
            )

            task_acq_opt = progress.add_task(description="Optimizing acquisition function", total=batch_size)
            self.acq_result, self.acq_value = optimize_acqf_discrete(
                acq_function=acq_func.ehvi,
                choices=candidate_X_t,
                q=batch_size,
                max_batch_size=self.max_batch_size,
                unique=True,
                exclude_points=training_X_t,  # 传入训练数据避免重复选择
                min_distance=1e-6,
                progress=progress,
                task=task_acq_opt,
            )

        if device.type == "cuda":
            best_samples = [res.cpu().numpy() for res in self.acq_result]
        else:
            best_samples = [res.numpy() for res in self.acq_result]
        # 推荐类型判断
        recommend_type = self._get_expoit_or_explore(self.acq_value)
        # 查找最优样本对应的候选点
        selected_indices = [np.argwhere(np.all(candidate_X == best_sample, axis=1)).flatten() for best_sample in best_samples]
        selected_indices = np.array(selected_indices).squeeze()
        selected_conditions = self.name_data[selected_indices].squeeze()
        # 计算预测值和置信度
        with torch.no_grad():
            posterior = self.global_model.posterior(self.acq_result)
            pred_mean = posterior.mean.cpu().numpy()  # (batch_size, num_objectives)
            pred_var = posterior.variance.cpu().numpy()  # (batch_size, num_objectives)
            pred_std = np.sqrt(pred_var)  # 标准差作为置信度

        # 对最大化目标的预测结果进行反变换（重新取负号）
        for i, d in enumerate(opt_direct_info):
            if d["opt_direct"] == "min":
                pred_mean[:, i] = -pred_mean[:, i]

        # 最终日志（用原 console 或 progress 的 console）
        self.opt_console.print("✅ Finish optimization", style="green")
        return selected_conditions, recommend_type, pred_mean, pred_std

    def _get_expoit_or_explore(self, acq_value):
        with torch.no_grad():
            posterior = self.global_model.posterior(self.acq_result)
            pred_mean = posterior.mean  # (batch_size, num_objectives)
            pred_var = posterior.variance  # (batch_size, num_objectives)
        # 计算每个点的HVI
        hvi_values = torch.tensor([compute_hvi(pred_mean[i], self.pareto_y, self.ref_point) for i in range(pred_mean.shape[0])])
        # EHVI已经在acq_value中返回（可能需调整形状）
        ehvi_values = acq_value.to(device="cpu")  # 确保形状为 (batch_size,)
        # 计算利用分数（Exploit Score）
        # TODO: need to change the defination here!!!
        exploit_scores = hvi_values / (ehvi_values + 1e-6)  # 避免除以0
        explore_scores = 1 - exploit_scores
        # 输出每个推荐点的探索-利用倾向
        for i in range(self.acq_result.shape[0]):
            self.opt_console.log(
                f"Point {i}: "
                f"EHVI = {ehvi_values[i]:.3f}, "
                f"HVI = {hvi_values[i]:.3f}, "
                f"Exploit Score = {exploit_scores[i]:.3f}, "
                f"Explore Score = {explore_scores[i]:.3f}"
            )
        return ["exploit" if exploit_scores[i] > explore_scores[i] else "explore" for i in range(self.acq_result.shape[0])]
