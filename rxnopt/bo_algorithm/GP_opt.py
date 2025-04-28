from typing import Tuple, List
import numpy as np
import torch
import gpytorch
from botorch.models import SingleTaskGP, ModelListGP
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
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

    def __init__(self, num_dims: int, device: str):
        super().__init__(num_dims)
        self.device = device
        self.model = None
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=GreaterThan(1e-5), noise_prior=GammaPrior(1.5, 0.1)).to(
            self.device
        )

    def fit(self, train_x: torch.Tensor, train_y: torch.Tensor) -> None:
        """Build and train a single GP model"""
        # Move input data to the correct device
        train_x = train_x.to(self.device)
        train_y = train_y.to(self.device)

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
        ).to(self.device)

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
        # Move input to the correct device
        x = x.to(self.device)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(x))
            mean = observed_pred.mean
            variance = observed_pred.variance
        return mean.cpu(), variance.cpu()  # Return results on CPU for consistency


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

    def __init__(
        self,
        model: ModelListGP,
        sampler: SobolQMCNormalSampler,
        ref_point: torch.Tensor,
        partitioning: NondominatedPartitioning,
        maximum_metrics: bool,
    ):
        super().__init__(model, sampler)
        self.ref_point = ref_point
        self.partitioning = partitioning
        if maximum_metrics:
            self.ehvi = qLogExpectedHypervolumeImprovement(
                model=model,
                sampler=sampler,
                ref_point=ref_point,
                partitioning=partitioning,
            )
        else:
            self.ehvi = qLogExpectedHypervolumeImprovement(
                model=model,
                sampler=sampler,
                ref_point=ref_point,
                partitioning=partitioning,
            )


class ParetoFrontCalculator:
    """Class for calculating Pareto fronts"""

    @staticmethod
    def calculate_target_function(points: np.ndarray) -> np.ndarray:
        """
        Calculate Pareto front for points in arbitrary dimensions

        Args:
            points: numpy array of shape (n_points, n_dimensions)

        Returns:
            numpy array of Pareto optimal points
        """
        if len(points) == 0:
            return np.array([])

        points = np.asarray(points)

        pareto_front = [points[0]]  # Initialize list of Pareto optimal points

        for point in points[1:]:
            is_pareto = True
            to_remove = []

            # Compare with all points in current Pareto front
            for i, pf_point in enumerate(pareto_front):
                # Check if the current point dominates any existing Pareto point
                if np.all(point >= pf_point) and np.any(point > pf_point):
                    to_remove.append(i)
                # Check if any existing Pareto point dominates the current point
                elif np.all(pf_point >= point) and np.any(pf_point > point):
                    is_pareto = False
                    break

            # Remove dominated points from Pareto front
            for i in reversed(to_remove):
                pareto_front.pop(i)

            # Add current point if it's Pareto optimal
            if is_pareto:
                pareto_front.append(point)

        return torch.tensor(pareto_front)
