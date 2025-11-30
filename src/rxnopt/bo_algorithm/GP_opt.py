from typing import Tuple, List
import numpy as np
import torch
import gpytorch
from botorch.models import SingleTaskGP

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
