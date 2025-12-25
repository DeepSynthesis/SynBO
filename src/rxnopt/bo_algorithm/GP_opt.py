from typing import Tuple, List
import numpy as np
import torch
import gpytorch
from botorch.models import SingleTaskGP, ModelListGP
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.sampling.normal import SobolQMCNormalSampler

from gpytorch.constraints import GreaterThan
from gpytorch.priors import GammaPrior
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood


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
        # Use edboplus-style likelihood initialization
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(GammaPrior(1.5, 0.1)).to(self.device)
        self.likelihood.noise = 0.5

    def fit(self, train_x: torch.Tensor, train_y: torch.Tensor) -> None:
        """Build and train a single GP model"""
        # Move input data to the correct device
        train_x = train_x.to(self.device)
        train_y = train_y.to(self.device)

        # Use adaptive covariance module configuration
        # Scale initial lengthscale based on input dimensionality
        initial_lengthscale = max(0.5, min(5.0, np.sqrt(self.num_dims)))

        covar_module = ScaleKernel(
            MaternKernel(
                ard_num_dims=self.num_dims,
                lengthscale_prior=GammaPrior(3.0, 1.0),  # More concentrated prior
            ),
            outputscale_prior=GammaPrior(2.0, 0.5),
        )
        # Set adaptive initial lengthscale
        covar_module.base_kernel.lengthscale = initial_lengthscale

        self.model = SingleTaskGP(
            train_X=train_x,
            train_Y=train_y,
            covar_module=covar_module,
        ).to(self.device)

        # Add noise constraint like edboplus
        self.model.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))

        # Train the model
        self.model.train()
        self.likelihood.train()

        # Improved training with learning rate schedule and early stopping
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)
        mll = ExactMarginalLogLikelihood(self.likelihood, self.model)

        best_loss = float("inf")
        patience_counter = 0
        patience_limit = 100

        for epoch in range(1000):
            optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y.squeeze(-1))
            loss.backward()
            optimizer.step()
            scheduler.step(loss.item())

            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience_limit:
                break

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
    """Enhanced Expected Hypervolume Improvement acquisition function with exploration bonus"""

    def __init__(
        self,
        model: ModelListGP,
        sampler: SobolQMCNormalSampler,
        ref_point: torch.Tensor,
        partitioning: NondominatedPartitioning,
        maximum_metrics: bool,
        exploration_weight: float = 0.1,  # New parameter for exploration
    ):
        super().__init__(model, sampler)
        self.ref_point = ref_point
        self.partitioning = partitioning
        self.exploration_weight = exploration_weight

        self.ehvi = qLogExpectedHypervolumeImprovement(
            model=model,
            sampler=sampler,
            ref_point=ref_point,
            partitioning=partitioning,
        )

    def evaluate_with_exploration(self, X: torch.Tensor) -> torch.Tensor:
        """Enhanced evaluation with exploration bonus"""
        # Standard EHVI
        ehvi_val = self.ehvi(X)

        # Add exploration bonus based on predictive variance
        with torch.no_grad():
            posterior = self.model.posterior(X)
            # Average variance across objectives as exploration signal
            exploration_bonus = torch.mean(posterior.variance, dim=-1)

        return ehvi_val + self.exploration_weight * exploration_bonus


class ParetoFrontCalculator:
    """Class for calculating Pareto fronts"""

    @staticmethod
    def calculate_target_function(points: np.ndarray, progress: object, task: object) -> np.ndarray:
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
            progress.update(task, advance=1)
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
        print(pareto_front)  # TODO: need to remove before release
        return torch.tensor(np.array(pareto_front))
