from typing import List
from loguru import logger
import numpy as np
import torch

from botorch.models import ModelListGP
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.sampling.normal import SobolQMCNormalSampler

from .utils.utils import compute_hvi

from .bo_algorithm.GP_opt import GPSurrogateModel, EHVIAcquisitionFunction, ParetoFrontCalculator
from .bo_algorithm.acf_opt import optimize_acqf_discrete


class Optimizer:
    def __init__(self, method, name_data, num_samples: int = 1024, seed: int = 1145141):
        self.name_data = name_data
        self.num_samples = num_samples
        self.seed = seed
        self.surrogate_model_class = GPSurrogateModel
        self.acquisition_function_class = EHVIAcquisitionFunction
        self.target_evaluator = ParetoFrontCalculator()

    def optimize(
        self,
        training_X: np.ndarray,
        training_y: np.ndarray,
        candidate_X: np.ndarray,
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
        training_y_dict = training_y.copy()
        if isinstance(training_y, dict):
            training_y = np.array(list(training_y.values())).T

        training_X_t = torch.tensor(training_X).double()
        training_y_t = torch.tensor(training_y).double()
        candidate_X_t = torch.tensor(candidate_X).double()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.double

        training_X_t = training_X_t.to(device=device, dtype=dtype)
        training_y_t = training_y_t.to(device=device, dtype=dtype)
        candidate_X_t = candidate_X_t.to(device=device, dtype=dtype)

        models = []
        for i in range(training_y_t.shape[1]):
            train_y_i = training_y_t[:, i].reshape(-1, 1)
            logger.info(f"Fitting previous data points for {list(training_y_dict.keys())[i]}...")
            model_i = self.surrogate_model_class(device=device, num_dims=training_X.shape[1])
            model_i.fit(training_X_t, train_y_i)
            models.append(model_i.model)
        # 多输出模型
        self.global_model = ModelListGP(*models)
        logger.info("Calculating Pareto frontiers...")
        self.pareto_y = self.target_evaluator.calculate_target_function(training_y).to(device=device)
        self.ref_point = torch.tensor([0.0] * training_y_t.shape[1]).to(device=device)  # 保持与模型一致的数据类型

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.num_samples]), seed=self.seed)  # 采样器直接生成GPU张量
        partitioning = NondominatedPartitioning(ref_point=self.ref_point, Y=self.pareto_y)  # 确保输入数据在GPU上
        acq_func = self.acquisition_function_class(
            model=self.global_model, sampler=sampler, ref_point=self.ref_point, partitioning=partitioning, maximum_metrics=maximum_metrics
        )

        logger.info("Optimizing acquisition function...")
        self.acq_result, self.acq_value = optimize_acqf_discrete(
            acq_function=acq_func.ehvi, choices=candidate_X_t, q=batch_size, unique=True, device=device
        )
        if device.type == "cuda":
            best_samples = [res.cpu().numpy() for res in self.acq_result]
        else:
            best_samples = [res.numpy() for res in self.acq_result]

        recommend_type = self._get_expoit_or_explore(self.acq_value)
        # Find closest candidate points to optimal samples
        selected_indices = [np.argwhere(np.all(candidate_X == best_sample, axis=1)).flatten() for best_sample in best_samples]
        selected_indices = np.array(selected_indices).squeeze()
        selected_conditions = self.name_data[selected_indices].squeeze()
        logger.info("Finish optimizerization")
        return selected_conditions, recommend_type

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
            print(
                f"Point {i}: "
                f"EHVI = {ehvi_values[i]:.3f}, "
                f"HVI = {hvi_values[i]:.3f}, "
                f"Exploit Score = {exploit_scores[i]:.3f}, "
                f"Explore Score = {explore_scores[i]:.3f}"
            )
        return ["exploit" if exploit_scores[i] > explore_scores[i] else "explore" for i in range(self.acq_result.shape[0])]
