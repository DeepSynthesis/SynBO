from typing import List, Tuple
import numpy as np
import torch
from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn, TimeRemainingColumn

from botorch.models import ModelListGP
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.sampling.normal import SobolQMCNormalSampler

from synbo.utils.logger import console
from synbo.algorithm.sg_model import (
    BNNEnsembleSurrogateModel,
    BayesianLinearSurrogateModel,
    GPSurrogateModel,
    RFSurrogateModel,
    SklearnModelWrapper,
)
from synbo.algorithm.acq_function import (
    EHVIAcquisitionFunction,
    NEIAcquisitionFunction,
    ParEGOAcquisitionFunction,
    UCBAcquisitionFunction,
    ParetoFrontCalculator,
)


class DefaultBO:
    def __init__(
        self,
        random_seed: int = 42,
        surrogate_model: str = "GP",
        acq_func: str = "EHVI",
        device: torch.device = torch.device("cpu"),
        accuracy: str = "medium",
    ):
        self.random_seed = random_seed
        self.console = console

        if accuracy == "medium":
            self.mc_num_samples, self.max_batch_size = 256, 2048
        self.device = device

        if surrogate_model == "GP":
            self.surrogate_model_class = GPSurrogateModel
        elif surrogate_model == "RF":
            self.surrogate_model_class = RFSurrogateModel
        elif surrogate_model == "BNN":
            self.surrogate_model_class = BNNEnsembleSurrogateModel
        elif surrogate_model == "BayesianLinear":
            self.surrogate_model_class = BayesianLinearSurrogateModel
        else:
            raise ValueError(f"Unknown surrogate model: {surrogate_model}")

        if acq_func == "EHVI":
            self.acquisition_function_class = EHVIAcquisitionFunction
        elif acq_func == "UCB":
            self.acquisition_function_class = UCBAcquisitionFunction
        elif acq_func == "ParEGO":
            self.acquisition_function_class = ParEGOAcquisitionFunction
        elif acq_func == "NEI":
            self.acquisition_function_class = NEIAcquisitionFunction
        else:
            raise ValueError(f"Unknown acquisition function: {acq_func}")

        self.target_evaluator = ParetoFrontCalculator()

    def optimize(
        self,
        training_X: np.ndarray,
        training_y: np.ndarray,
        candidate_X: np.ndarray,
        opt_metric_settings: List[dict],
        batch_size: int,
        done_name: np.ndarray = None,
        candidate_name: np.ndarray = None,
        condition_types: List[str] = None,
    ) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:

        training_X_t = torch.tensor(training_X).double().to(device=self.device)
        training_y_t = torch.tensor(training_y).double()
        training_y_t = self._weight_y(training_y_t, opt_metric_settings).to(device=self.device)
        candidate_X_t = torch.tensor(candidate_X).double()
        # constraint_mask_t = torch.tensor(constraints) if constraints is not None else None

        with Progress(
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            num_models = training_y_t.shape[1]
            task_train = progress.add_task(description="Training surrogate models", total=num_models, start=True)

            models = []
            for i in range(num_models):
                progress.log(f"Fitting model for {i+1}th model...", style="yellow")

                # Instantiate model with random_seed for reproducibility
                if self.surrogate_model_class in [RFSurrogateModel, BNNEnsembleSurrogateModel]:
                    model_i = self.surrogate_model_class(device=self.device, num_dims=training_X_t.shape[1], random_seed=self.random_seed)
                else:
                    model_i = self.surrogate_model_class(device=self.device, num_dims=training_X_t.shape[1])

                if isinstance(model_i, GPSurrogateModel):
                    model_i.fit(training_X_t, training_y_t[:, i].unsqueeze(-1))
                    models.append(model_i.model)
                else:
                    wrapper = SklearnModelWrapper(model_i)
                    wrapper.fit_surrogate(training_X_t, training_y_t[:, i].unsqueeze(-1))
                    models.append(wrapper)

                progress.update(task_train, advance=1)

            self.global_model = ModelListGP(*models)

            task_pareto = progress.add_task(description="Calculating Pareto frontiers", total=len(training_y) - 1)
            training_y_np = training_y_t.cpu().numpy()
            self.pareto_y = self.target_evaluator.calculate_target_function(training_y_np, progress, task_pareto).to(device=self.device)

            # Dynamically calculate reference points based on training data
            y_min = training_y_t.min(dim=0).values
            y_max = training_y_t.max(dim=0).values
            y_range = y_max - y_min

            # Reference point = min - 10% range (ensure reference point is below/worse than Pareto front)
            ref_point_values = []
            for i, omi in enumerate(opt_metric_settings):
                if y_range[i] > 0:
                    ref_val = y_min[i]  # - 0.1 * y_range[i]
                else:
                    # If all values are the same, give a small offset
                    ref_val = y_min[i] - 0.1
                ref_point_values.append(ref_val)

            self.ref_point = torch.tensor(ref_point_values, dtype=torch.double, device=self.device)

            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.mc_num_samples]), seed=self.random_seed)
            
            # Check number of objectives
            num_objectives = len(opt_metric_settings)
            
            if self.acquisition_function_class == EHVIAcquisitionFunction:
                if num_objectives > 1:
                    # Multi-objective: use EHVI with ModelListGP
                    partitioning = NondominatedPartitioning(ref_point=self.ref_point, Y=self.pareto_y)
                    acq_func = self.acquisition_function_class(
                        model=self.global_model,
                        sampler=sampler,
                        device=self.device,
                        ref_point=self.ref_point,
                        partitioning=partitioning,
                        train_x=torch.Tensor(training_X).to(self.device),
                        num_objectives=num_objectives,  # Pass num_objectives for correct acq function selection
                    )
                else:
                    # Single-objective: use qExpectedImprovement directly
                    # Get the best value from normalized training data
                    best_value = training_y_t.max(dim=0).values
                    from botorch.acquisition.monte_carlo import qExpectedImprovement
                    # Use individual GP model (not ModelListGP for single objective)
                    single_model = models[0]
                    acq_func = qExpectedImprovement(
                        model=single_model,
                        best_f=best_value,
                        sampler=sampler,
                    )
            elif self.acquisition_function_class == UCBAcquisitionFunction:
                weights = torch.tensor([d["metric_weight"] for d in opt_metric_settings], dtype=torch.double, device=self.device)
                acq_func = self.acquisition_function_class(
                    model=self.global_model,
                    sampler=sampler,
                    beta=2.0,
                    weights=weights,
                    device=self.device,
                )
            elif self.acquisition_function_class == ParEGOAcquisitionFunction:

                acq_func = self.acquisition_function_class(
                    model=self.global_model,
                    sampler=sampler,
                    device=self.device,
                    X_baseline=training_X_t,
                    num_objectives=len(opt_metric_settings),
                )
            elif self.acquisition_function_class == NEIAcquisitionFunction:
                weights = torch.tensor([d["metric_weight"] for d in opt_metric_settings], dtype=torch.double, device=self.device)
                acq_func = self.acquisition_function_class(
                    model=self.global_model, sampler=sampler, device=self.device, X_baseline=training_X_t, weights=weights
                )
            else:
                raise ValueError(f"Unknown acquisition function class: {self.acquisition_function_class}")

            # candidate_X_t = candidate_X_t[constraint_mask_t] if constraint_mask_t is not None else candidate_X_t

            # TODO: need to Re-implementate reagent boost mechanism.

            task_acq_opt = progress.add_task(description="Optimizing acquisition function", total=batch_size)
            self.acq_result, self.acq_value = acq_func.optimize_acqf_discrete(
                q=batch_size,
                choices=candidate_X_t,
                max_batch_size=self.max_batch_size,
                unique=True,
                progress=progress,
            )

        if self.device.type == "cuda":
            best_samples = [res.cpu().numpy() for res in self.acq_result]
        else:
            best_samples = [res.numpy() for res in self.acq_result]

        recommend_type = ["Unknown"] * batch_size  # self._get_exploit_or_explore()
        pred_mean, pred_std = self._get_predictions()

        for i, d in enumerate(opt_metric_settings):
            if d["opt_direct"] == "min":
                pred_mean[:, i] = -pred_mean[:, i]

        pred_mean = self._unweight_y(torch.tensor(pred_mean), opt_metric_settings).numpy()
        pred_std = self._unweight_y(torch.tensor(pred_std), opt_metric_settings).numpy()
        return best_samples, recommend_type, pred_mean, pred_std

    def _get_predictions(self) -> Tuple[np.ndarray, np.ndarray]:
        self.acq_result = self.acq_result.to(device=self.device)

        with torch.no_grad():
            posterior = self.global_model.posterior(self.acq_result)
            pred_mean = posterior.mean
            pred_var = posterior.variance

        # Handle different shapes from posterior
        if pred_mean.dim() == 2:
            # Already in correct shape (n_points, n_outputs)
            pass
        elif pred_mean.dim() == 3:
            # Shape: (1, n_points, n_outputs) -> (n_points, n_outputs)
            pred_mean = pred_mean.squeeze(0)
            pred_var = pred_var.squeeze(0)

        # Ensure correct shapes
        if pred_mean.dim() != 2:
            raise ValueError(f"Unexpected pred_mean shape: {pred_mean.shape}")

        pred_mean = pred_mean.cpu().numpy()
        pred_var = pred_var.cpu().numpy()
        pred_std = np.sqrt(pred_var)
        return pred_mean, pred_std

    def _weight_y(self, training_y: torch.Tensor, opt_metric_settings: List[dict]) -> torch.Tensor:
        weights = torch.tensor([d["metric_weight"] for d in opt_metric_settings])
        training_y = training_y * weights
        return training_y

    def _unweight_y(self, training_y: torch.Tensor, opt_metric_settings: List[dict]) -> torch.Tensor:
        weights = torch.tensor([d["metric_weight"] for d in opt_metric_settings])
        training_y = training_y / weights
        return training_y
