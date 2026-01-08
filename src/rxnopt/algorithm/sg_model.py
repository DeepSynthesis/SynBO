from typing import Tuple, List
import numpy as np
import torch
import gpytorch
from botorch.models import SingleTaskGP

from gpytorch.constraints import GreaterThan
from gpytorch.priors import GammaPrior
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge


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


class RFSurrogateModel(BaseSurrogateModel):
    """
    Random Forest Surrogate Model.
    Estimates uncertainty by calculating the variance across individual decision trees.
    """

    def __init__(self, num_dims: int, device: str = "cpu", n_estimators: int = 100):
        super().__init__(num_dims)
        # Random Forest usually runs on CPU via sklearn, but we keep the device param for API consistency
        self.device = device
        self.n_estimators = n_estimators
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators, min_samples_leaf=3, random_state=42  # Prevent overfitting to single points
        )

    def fit(self, train_x: torch.Tensor, train_y: torch.Tensor) -> None:
        """Fit the Random Forest model using sklearn"""
        # Convert to numpy and move to CPU
        X_np = train_x.cpu().detach().numpy()
        # Flatten y: (N, 1) -> (N,)
        y_np = train_y.cpu().detach().numpy().ravel()

        self.model.fit(X_np, y_np)

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict mean and variance.
        Variance is estimated empirically from the predictions of individual trees.
        """
        X_np = x.cpu().detach().numpy()

        # Get predictions from every individual tree in the forest
        # Shape: (n_estimators, n_samples)
        tree_predictions = np.stack([tree.predict(X_np) for tree in self.model.estimators_])

        # Calculate mean and variance across trees
        mean = np.mean(tree_predictions, axis=0)
        variance = np.var(tree_predictions, axis=0)

        # Add a small epsilon to variance to prevent numerical issues in acquisition functions
        variance = np.maximum(variance, 1e-6)

        # Convert back to torch tensors
        mean_tensor = torch.from_numpy(mean).float()
        # Ensure shape is (N, 1) to match GP output
        if mean_tensor.dim() == 1:
            mean_tensor = mean_tensor.unsqueeze(-1)

        var_tensor = torch.from_numpy(variance).float()
        if var_tensor.dim() == 1:
            var_tensor = var_tensor.unsqueeze(-1)

        return mean_tensor, var_tensor


class BNNEnsembleSurrogateModel(BaseSurrogateModel):
    """
    Deep Ensemble Surrogate Model (Approximation of BNN).
    Trains multiple MLPs and uses the ensemble statistics for mean and variance.
    """

    def __init__(self, num_dims: int, device: str, n_models: int = 5, hidden_dim: int = 64):
        super().__init__(num_dims)
        self.device = device
        self.n_models = n_models
        self.hidden_dim = hidden_dim
        self.models = []  # List to store individual networks

    def _create_mlp(self):
        """Create a simple MLP suitable for low-data regimes"""
        return nn.Sequential(
            nn.Linear(self.num_dims, self.hidden_dim),
            nn.Tanh(),  # Tanh or ReLU usually works
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1),
        ).to(self.device)

    def fit(self, train_x: torch.Tensor, train_y: torch.Tensor) -> None:
        """Train multiple independent neural networks"""
        train_x = train_x.to(self.device)
        train_y = train_y.to(self.device)

        self.models = []  # Reset models

        for i in range(self.n_models):
            model = self._create_mlp()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            criterion = nn.MSELoss()

            model.train()
            # Simple training loop for each model
            # In production, you might want batching, but full-batch is fine for BO (small data)
            for epoch in range(200):
                optimizer.zero_grad()
                output = model(train_x)
                loss = criterion(output, train_y)
                loss.backward()
                optimizer.step()

            model.eval()
            self.models.append(model)

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict using the ensemble statistics"""
        x = x.to(self.device)

        predictions = []
        with torch.no_grad():
            for model in self.models:
                # Output shape (N, 1)
                predictions.append(model(x))

        # Stack shape: (n_models, N, 1)
        predictions = torch.stack(predictions)

        # Calculate Mean and Variance across the ensemble dimension (dim=0)
        mean = torch.mean(predictions, dim=0)
        variance = torch.var(predictions, dim=0)

        # Add epsilon for numerical stability
        variance = torch.maximum(variance, torch.tensor(1e-6, device=self.device))

        return mean.cpu(), variance.cpu()


class BayesianLinearSurrogateModel(BaseSurrogateModel):
    """
    Bayesian Linear Regression Surrogate Model.
    Uses sklearn's BayesianRidge. Fast and robust, but limited expressiveness.
    """

    def __init__(self, num_dims: int, device: str = "cpu"):
        super().__init__(num_dims)
        self.device = device  # Kept for compatibility
        self.model = BayesianRidge(n_iter=300, tol=1e-3, fit_intercept=True, compute_score=True)

    def fit(self, train_x: torch.Tensor, train_y: torch.Tensor) -> None:
        X_np = train_x.cpu().detach().numpy()
        y_np = train_y.cpu().detach().numpy().ravel()

        # Fit the Bayesian Ridge model
        self.model.fit(X_np, y_np)

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        X_np = x.cpu().detach().numpy()

        # Sklearn returns standard deviation if return_std=True
        mean_np, std_np = self.model.predict(X_np, return_std=True)

        # Convert std to variance
        variance_np = std_np**2

        # Convert to tensors and ensure shape (N, 1)
        mean_tensor = torch.from_numpy(mean_np).float().unsqueeze(-1)
        var_tensor = torch.from_numpy(variance_np).float().unsqueeze(-1)

        return mean_tensor, var_tensor
