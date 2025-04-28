from typing import List
import numpy as np
import torch
from botorch.optim import optimize_acqf_discrete

from botorch.models import ModelListGP
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.optim import optimize_acqf_discrete
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.sampling.normal import SobolQMCNormalSampler

from .bo_algorithm.GP_opt import GPSurrogateModel, EHVIAcquisitionFunction, ParetoFrontCalculator


class Optimizer:
    def __init__(self, method="default", num_samples: int = 1024, seed: int = 1145141):
        self.num_samples = num_samples
        self.seed = seed
        self.surrogate_model_class = GPSurrogateModel
        self.acquisition_function_class = EHVIAcquisitionFunction
        self.target_evaluator = ParetoFrontCalculator()

    def optimize(
        self, training_X: np.ndarray, training_y: np.ndarray, candidate_X: np.ndarray, batch_size: int = 5, opt_weights: dict = None
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
        if isinstance(training_y, dict):
            training_y = np.array(list(training_y.values())).T

        training_X_t = torch.tensor(training_X).double()
        candidate_X_t = torch.tensor(candidate_X).double()

        # Build GP models for each objective
        models = []
        for i in range(training_y.shape[1]):
            # Get single objective
            train_y_i = training_y[:, i].reshape(-1, 1)
            train_y_i_t = torch.tensor(train_y_i).double()

            # Build and train models
            model_i = self.surrogate_model_class(num_dims=training_X.shape[1])
            model_i.fit(training_X_t, train_y_i_t)
            models.append(model_i.model)

        # Create multi-output model
        global_model = ModelListGP(*models)

        # Calculate Pareto front and reference point
        pareto_y = self.target_evaluator.calculate_target_function(training_y)
        ref_point = torch.tensor(np.min(training_y, axis=0)).float()

        # Set up acquisition function
        sampler = SobolQMCNormalSampler(torch.Size([self.num_samples]), seed=self.seed)
        from IPython import embed; embed()
        partitioning = NondominatedPartitioning(ref_point=ref_point, Y=torch.tensor(pareto_y).float())

        acq_func = self.acquisition_function_class(model=global_model, sampler=sampler, ref_point=ref_point, partitioning=partitioning)

        # Optimize acquisition function
        acq_result = optimize_acqf_discrete(acq_function=acq_func.ehvi, choices=candidate_X_t, q=batch_size, unique=True)

        # Find closest candidate points to optimal samples
        best_samples = acq_result[0].detach().cpu().numpy()
        selected_indices = []

        for sample in best_samples:
            d_i = np.linalg.norm(candidate_X - sample, axis=1)
            a = np.argmin(d_i)
            selected_indices.append(a)

        return selected_indices
