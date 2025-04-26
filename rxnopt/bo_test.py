import torch
import numpy as np
import gpytorch
from torch.nn.utils import clip_grad_norm_
from gpytorch.models import ExactGP
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.priors import GammaPrior
from gpytorch.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP, ModelListGP
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.optim import optimize_acqf_discrete
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.constraints import Positive


class BayesianOptimizer:
    def __init__(self, num_samples=1024, seed=1145141):
        self.num_samples = num_samples
        self.seed = seed

    def optimize(self, train_x, train_y, candidate_x, batch_size=5):
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

        if type(train_y) == dict:
            train_y = np.array(list(train_y.values())).T

        # from IPython import embed; embed()
        train_x_t = torch.tensor(train_x).double()
        train_y_t = torch.tensor(train_y).double()
        candidate_x_t = torch.tensor(candidate_x).double()

        # Build GP models for each objective
        models = []
        for i in range(train_y.shape[1]):
            # Get single objective
            train_y_i = train_y[:, i].reshape(-1, 1)
            train_y_i_t = torch.tensor(train_y_i).double()

            # Build and train GP model
            model_i = self._build_single_model(train_x_t, train_y_i_t)
            models.append(model_i)

        # Create multi-output model
        bigmodel = ModelListGP(*models)

        # Calculate Pareto front and reference point
        pareto_y = self._calculate_pareto_front(train_y)
        ref_point = torch.tensor(np.min(train_y, axis=0)).float()

        # Set up acquisition function
        sampler = SobolQMCNormalSampler(torch.Size([self.num_samples]), seed=self.seed)

        partitioning = NondominatedPartitioning(ref_point=ref_point, Y=torch.tensor(pareto_y).float())

        EHVI = qExpectedHypervolumeImprovement(
            model=bigmodel,
            sampler=sampler,
            ref_point=ref_point,
            partitioning=partitioning,
        )

        # Optimize acquisition function
        acq_result = optimize_acqf_discrete(acq_function=EHVI, choices=candidate_x_t, q=batch_size, unique=True)

        # Find closest candidate points to optimal samples
        best_samples = acq_result[0].detach().cpu().numpy()
        selected_indices = []

        for sample in best_samples:
            d_i = np.linalg.norm(candidate_x - sample, axis=1)
            a = np.argmin(d_i)
            selected_indices.append(a)

        return selected_indices

    def _build_single_model(self, train_x, train_y):
        """Build and train a single GP model"""
        likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=GreaterThan(1e-5), noise_prior=GammaPrior(1.5, 0.1))

        model = SingleTaskGP(
            train_X=train_x,
            train_Y=train_y,
            covar_module=ScaleKernel(
                MaternKernel(
                    ard_num_dims=train_x.shape[1],
                    lengthscale_prior=GammaPrior(2.0, 0.2),
                    lengthscale_constraint=Positive(),  # 强制 lengthscale > 0
                ),
                outputscale_prior=GammaPrior(5.0, 0.5),
                outputscale_constraint=Positive(),  # 强制 outputscale > 0
            ),
        )

        # Train the model
        model.train()
        likelihood.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = ExactMarginalLogLikelihood(likelihood, model)

        for _ in range(1000):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y.squeeze(-1))
            loss.backward()
            
            # 梯度裁剪防止爆炸
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 可选：手动约束参数
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if 'lengthscale' in name or 'noise' in name:
                        param.clamp_(min=1e-6)

        model.eval()
        likelihood.eval()

        return model

    def _calculate_pareto_front(self, points):
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
