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
        self.likelihood.noise = 5.0

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
                lengthscale_prior=GammaPrior(2.0, 0.2),  # More concentrated prior
            ),
            outputscale_prior=GammaPrior(5.0, 0.5),
        )
        # Set adaptive initial lengthscale
        covar_module.base_kernel.lengthscale = initial_lengthscale

        self.model = SingleTaskGP(
            train_X=train_x,
            train_Y=train_y,
            covar_module=covar_module,
            likelihood=self.likelihood,
        ).to(self.device)

        # Add noise constraint
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
        patience_limit = 1000

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

    def __init__(self, num_dims: int, device: str = "cpu", n_estimators: int = 200, random_seed: int = 42):
        super().__init__(num_dims)
        # Random Forest usually runs on CPU via sklearn, but we keep the device param for API consistency
        self.device = device
        self.n_estimators = n_estimators
        self.random_seed = random_seed
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators, min_samples_leaf=10, random_state=random_seed  # Prevent overfitting to single points
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

    def __init__(self, num_dims: int, device: str, n_models: int = 10, hidden_dim: int = 256, random_seed: int = 42):
        super().__init__(num_dims)
        self.device = device
        self.n_models = n_models
        self.hidden_dim = hidden_dim
        self.random_seed = random_seed
        self.models = []  # List to store individual networks

    def _create_mlp(self, seed: int):
        """Create a simple MLP suitable for low-data regimes with deterministic initialization"""
        # Set seed for reproducible weight initialization
        torch.manual_seed(seed)
        return (
            nn.Sequential(
                nn.Linear(self.num_dims, self.hidden_dim),
                nn.Tanh(),  # Tanh or ReLU usually works
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_dim, 1),
            )
            .to(self.device)
            .double()
        )  # Convert to double precision

    def fit(self, train_x: torch.Tensor, train_y: torch.Tensor) -> None:
        """Train multiple independent neural networks"""
        train_x = train_x.to(self.device)
        train_y = train_y.to(self.device)

        self.models = []  # Reset models

        for i in range(self.n_models):
            # Use different but deterministic seed for each model
            model_seed = self.random_seed + i
            model = self._create_mlp(seed=model_seed)
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
        # Convert input to double to match the model dtype
        x = x.double()

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
        variance = torch.maximum(variance, torch.tensor(1e-6, device=self.device, dtype=variance.dtype))

        # Return as double to maintain precision
        return mean.cpu(), variance.cpu()


class BayesianLinearSurrogateModel(BaseSurrogateModel):
    """
    Bayesian Linear Regression Surrogate Model.
    Uses sklearn's BayesianRidge. Fast and robust, but limited expressiveness.
    """

    def __init__(self, num_dims: int, device: str = "cpu"):
        super().__init__(num_dims)
        self.device = device  # Kept for compatibility
        self.model = BayesianRidge(max_iter=300, tol=1e-3, fit_intercept=True, compute_score=True)

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


class SklearnModelWrapper(gpytorch.models.ExactGP):
    """
    BoTorch-compatible wrapper for sklearn models to make them compatible with ModelListGP.
    This allows RF and other sklearn models to be used in the BoTorch acquisition functions.
    """

    def __init__(self, surrogate_model: BaseSurrogateModel):
        # Create dummy likelihood and mean module for BoTorch compatibility
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super().__init__(None, None, likelihood)

        self.surrogate_model = surrogate_model
        self._is_fitted = False

        # Dummy mean and covariance modules to satisfy gpytorch requirements
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.RBFKernel()

        # Add batch_shape attribute for ModelListGP compatibility
        self._batch_shape = torch.Size([])

    def forward(self, x):
        # This method is required by gpytorch but won't be used for inference
        # since we override posterior()
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    @property
    def batch_shape(self):
        """Return batch_shape for ModelListGP compatibility"""
        return self._batch_shape

    def posterior(self, X, output_indices=None, observation_noise=False, posterior_transform=None):
        """
        Override the posterior method to use our sklearn model for predictions.
        This is what acquisition functions will call.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calling posterior")

        # Handle different input shapes - ensure 2D for sklearn
        if X.dim() == 3:
            # Batched input: (batch_size, q, d) -> (batch_size * q, d)
            batch_size, q, d = X.shape
            X_2d = X.view(-1, d)
        elif X.dim() == 2:
            # Regular input: (n, d)
            batch_size = None
            q = X.shape[0]
            d = X.shape[1]
            X_2d = X
        else:
            raise ValueError(f"Unsupported input shape: {X.shape}")

        # Get predictions from our surrogate model
        with torch.no_grad():
            mean, variance = self.surrogate_model.predict(X_2d)

        # Ensure we're on the right device and correct dtype
        if X.device != mean.device:
            mean = mean.to(X.device)
            variance = variance.to(X.device)
        mean = mean.to(X.dtype)
        variance = variance.to(X.dtype)

        # Create a posterior distribution compatible with BoTorch
        from gpytorch.distributions import MultivariateNormal
        from linear_operator.operators import DiagLinearOperator

        # Ensure variance is positive
        variance = torch.clamp(variance, min=1e-6)

        # Reshape to match GP output format
        if batch_size is not None:
            # For 3D input (batch_size, q, d), output should be (batch_size, q, 1)
            mean = mean.view(batch_size, q, 1)
            variance = variance.view(batch_size, q, 1)
            # Squeeze last dimension for MVN: (batch_size, q)
            mean_squeezed = mean.squeeze(-1)
            var_squeezed = variance.squeeze(-1)
        else:
            # For 2D input (n, d), output should be (n,)
            mean_squeezed = mean.squeeze(-1)
            var_squeezed = variance.squeeze(-1)

        # Create diagonal covariance
        covar = DiagLinearOperator(var_squeezed)

        # Create the posterior distribution
        mvn = MultivariateNormal(mean_squeezed, covar)

        # Apply posterior transform if provided
        if posterior_transform is not None:
            mvn = posterior_transform(mvn)

        # Wrap in BoTorch's GPyTorchPosterior
        from botorch.posteriors.gpytorch import GPyTorchPosterior

        return GPyTorchPosterior(mvn)

    def fit_surrogate(self, train_x: torch.Tensor, train_y: torch.Tensor):
        """Fit the underlying surrogate model"""
        self.surrogate_model.fit(train_x, train_y)
        self._is_fitted = True
