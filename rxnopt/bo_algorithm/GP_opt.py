from typing import Tuple, List
import numpy as np
import torch
import gpytorch
from botorch.models import SingleTaskGP, ModelListGP
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.sampling.normal import SobolQMCNormalSampler

from gpytorch.constraints import GreaterThan, Positive
from gpytorch.priors import GammaPrior
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.nn.utils import clip_grad_norm_


class BaseSurrogateModel:
    """Base class for surrogate models"""

    def __init__(self, num_dims: int):
        self.num_dims = num_dims

    def fit(self, train_x: torch.Tensor, train_y: torch.Tensor) -> None:
        raise NotImplementedError

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class GPSurrogateModel(BaseSurrogateModel):
    """Gaussian Process surrogate model implementation"""

    def __init__(self, num_dims: int):
        super().__init__(num_dims)
        self.model = None
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=GreaterThan(1e-5), noise_prior=GammaPrior(1.5, 0.1))

    def fit(self, train_x: torch.Tensor, train_y: torch.Tensor) -> None:
        """Build and train a single GP model"""

        self.model = SingleTaskGP(
            train_X=train_x,
            train_Y=train_y,
            covar_module=ScaleKernel(
                MaternKernel(
                    ard_num_dims=self.num_dims,
                    lengthscale_prior=GammaPrior(2.0, 0.2),
                    lengthscale_constraint=Positive(),
                ),
                outputscale_prior=GammaPrior(5.0, 0.5),
                outputscale_constraint=Positive(),
            ),
        )

        # Train the model
        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        mll = ExactMarginalLogLikelihood(self.likelihood, self.model)

        for _ in range(1000):
            optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y.squeeze(-1))
            loss.backward()

            # Gradient clipping
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()

            # Manual parameter constraints
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if "lengthscale" in name or "noise" in name:
                        param.clamp_(min=1e-6)

        self.model.eval()
        self.likelihood.eval()

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Make predictions with the GP model"""
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(x))
            mean = observed_pred.mean
            variance = observed_pred.variance
        return mean, variance


class BaseAcquisitionFunction:
    """Base class for acquisition functions"""

    def __init__(self, model: ModelListGP, sampler: SobolQMCNormalSampler):
        self.model = model
        self.sampler = sampler

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the acquisition function"""
        raise NotImplementedError


class EHVIAcquisitionFunction(BaseAcquisitionFunction):
    """Expected Hypervolume Improvement acquisition function"""

    def __init__(self, model: ModelListGP, sampler: SobolQMCNormalSampler, ref_point: torch.Tensor, partitioning: NondominatedPartitioning):
        super().__init__(model, sampler)
        self.ref_point = ref_point
        self.partitioning = partitioning
        self.ehvi = qExpectedHypervolumeImprovement(
            model=model,
            sampler=sampler,
            ref_point=ref_point,
            partitioning=partitioning,
        )

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the EHVI acquisition function"""
        return self.ehvi(x.unsqueeze(-2))


class ParetoFrontCalculator:
    """Class for calculating Pareto fronts"""

    @staticmethod
    def calculate_2d_pareto_front(points: np.ndarray) -> np.ndarray:
        """Calculate 2D Pareto front"""
        sort_points = points.copy()
        sort_points = sorted(sort_points, key=lambda x: (x[0], x[1]), reverse=True)
        max_y = -1e9
        ans_points = []
        for x in sort_points:
            if x[1] > max_y:
                max_y = x[1]
                ans_points.append(x)
        ans_points.reverse()
        return np.array(ans_points)
