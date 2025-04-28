from typing import List
from loguru import logger
import numpy as np
import torch

from botorch.models import ModelListGP
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.sampling.normal import SobolQMCNormalSampler

from .bo_algorithm.GP_opt import GPSurrogateModel, EHVIAcquisitionFunction, ParetoFrontCalculator
from .bo_algorithm.acf_opt import optimize_acqf_discrete


class Optimizer:
    def __init__(self, method="default", num_samples: int = 1024, seed: int = 1145141):
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

        # Build GP models for each objective
        # 确定设备（自动选择GPU或CPU）
        dtype = torch.double  # 建议使用double精度避免数值问题
        # --- 1. 数据迁移到GPU ---
        training_X_t = training_X_t.to(device=device, dtype=dtype)
        training_y_t = training_y_t.to(device=device, dtype=dtype)
        # candidate_X_t = candidate_X_t.to(device=device, dtype=dtype)
        # --- 2. 构建GPU兼容的GP模型 ---
        models = []

        for i in range(training_y_t.shape[1]):
            train_y_i = training_y_t[:, i].reshape(-1, 1)

            # 模型初始化并迁移到GPU
            logger.info(f"Fitting previous data points for {list(training_y_dict.keys())[i]}...")
            model_i = self.surrogate_model_class(device=device, num_dims=training_X.shape[1])  # .to(device=device, dtype=dtype)
            model_i.fit(training_X_t, train_y_i)  # 数据已在GPU上
            models.append(model_i.model)
        # 多输出模型
        global_model = ModelListGP(*models).to(device="cpu")
        # --- 3. Pareto前沿和参考点（GPU兼容） ---
        logger.info("Calculating Pareto frontiers...")
        pareto_y = self.target_evaluator.calculate_target_function(training_y).to_device(device)
        ref_point = torch.tensor([0.0] * training_y_t.shape[1]).to_device(device)  # 保持与模型一致的数据类型
        # --- 4. 采样器和分区（GPU兼容） ---
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.num_samples]), seed=self.seed)  # 采样器直接生成GPU张量
        partitioning = NondominatedPartitioning(ref_point=ref_point, Y=torch.tensor(pareto_y))  # 确保输入数据在GPU上
        # --- 5. 采集函数（GPU兼容） ---
        acq_func = self.acquisition_function_class(
            model=global_model, sampler=sampler, ref_point=ref_point, partitioning=partitioning, maximum_metrics=maximum_metrics
        )
        # --- 6. 优化采集函数（GPU加速） ---
        logger.info("Optimizing acquisition function...")
        acq_result, acq_value = optimize_acqf_discrete(
            acq_function=acq_func.ehvi,
            choices=candidate_X_t,
            q=batch_size,
            unique=True,
        )
        print(acq_result, acq_value)
        # from IPython import embed

        # embed()
        # exit()
        # 结果迁移回CPU（如需）
        if device.type == "cuda":
            acq_result = tuple(res.cpu() for res in acq_result)
        # Find closest candidate points to optimal samples
        best_samples = acq_result[0].detach().cpu().numpy()
        selected_indices = []

        for sample in best_samples:
            d_i = np.linalg.norm(candidate_X - sample, axis=1)
            a = np.argmin(d_i)
            selected_indices.append(a)

        return selected_indices
