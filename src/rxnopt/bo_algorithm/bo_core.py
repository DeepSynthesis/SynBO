from typing import List, Tuple
import numpy as np
import torch
from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn, TimeRemainingColumn

from botorch.models import ModelListGP
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.sampling.normal import SobolQMCNormalSampler

from rxnopt.utils.util_func import compute_hvi
from rxnopt.utils.logger import console
from rxnopt.bo_algorithm.GP_opt import GPSurrogateModel
from rxnopt.bo_algorithm.acf_opt import optimize_acqf_discrete, EHVIAcquisitionFunction, ParetoFrontCalculator

import warnings
from linear_operator.utils.cholesky import NumericalWarning

warnings.filterwarnings("ignore", category=NumericalWarning)


class DefaultBO:
    def __init__(
        self,
        random_seed: int = 42,
        mc_num_samples: int = 64,
        max_batch_size: int = 128,
        surrogate_model: str = "GP",
        acq_func: str = "EHVI",
        target_eval: str = "Pareto",
        device: torch.device = torch.device("cpu"),
    ):
        self.random_seed = random_seed
        self.console = console
        self.mc_num_samples = mc_num_samples
        self.max_batch_size = max_batch_size
        self.device = device

        if surrogate_model == "GP":
            self.surrogate_model_class = GPSurrogateModel
        else:
            raise ValueError(f"Unknown surrogate model: {surrogate_model}")

        if acq_func == "EHVI":
            self.acquisition_function_class = EHVIAcquisitionFunction
        else:
            raise ValueError(f"Unknown acquisition function: {acq_func}")

        if target_eval == "Pareto":
            self.target_evaluator = ParetoFrontCalculator()
        else:
            raise ValueError(f"Unknown target evaluator: {target_eval}")

    def optimize(
        self,
        training_X: np.ndarray,
        training_y: np.ndarray,
        candidate_X: np.ndarray,
        opt_metric_settings: List[dict],
        batch_size: int,
        training_y_dict: dict,
    ) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:

        training_X_t = torch.tensor(training_X).double().to(device=self.device)
        training_y_t = torch.tensor(training_y).double()
        training_y_t = self._weight_y(training_y_t, opt_metric_settings).to(device=self.device)
        candidate_X_t = torch.tensor(candidate_X).double().to(device=self.device)

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
                key = list(training_y_dict.keys())[i]
                progress.log(f"Fitting model for [bold]{key}[/bold]...", style="yellow")

                train_y_i = training_y_t[:, i].reshape(-1, 1)
                model_i = self.surrogate_model_class(device=device, num_dims=training_X_t.shape[1])
                model_i.fit(training_X_t, train_y_i)
                models.append(model_i.model)
                progress.update(task_train, advance=1)

            self.global_model = ModelListGP(*models)

            task_pareto = progress.add_task(description="Calculating Pareto frontiers", total=len(training_y) - 1)
            training_y_np = training_y_t.cpu().numpy()
            self.pareto_y = self.target_evaluator.calculate_target_function(training_y_np, progress, task_pareto).to(device=self.device)

            self.ref_point = torch.tensor(
                [-1 if omi["opt_direct"] == "min" else 0 for omi in opt_metric_settings], dtype=float, device=self.device
            )

            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.mc_num_samples]), seed=self.random_seed)
            partitioning = NondominatedPartitioning(ref_point=self.ref_point, Y=self.pareto_y)
            acq_func = self.acquisition_function_class(
                model=self.global_model,
                sampler=sampler,
                ref_point=self.ref_point,
                partitioning=partitioning,
            )

            task_acq_opt = progress.add_task(description="Optimizing acquisition function", total=batch_size)
            self.acq_result, self.acq_value = optimize_acqf_discrete(
                acq_function=acq_func.ehvi,
                choices=candidate_X_t,
                q=batch_size,
                max_batch_size=self.max_batch_size,
                unique=True,
                exclude_points=training_X_t,
                min_distance=1e-6,
                progress=progress,
                task=task_acq_opt,
            )

        if self.device.type == "cuda":
            best_samples = [res.cpu().numpy() for res in self.acq_result]
        else:
            best_samples = [res.numpy() for res in self.acq_result]

        recommend_type = self._get_exploit_or_explore()
        pred_mean, pred_std = self._get_predictions()

        for i, d in enumerate(opt_metric_settings):
            if d["opt_direct"] == "min":
                pred_mean[:, i] = -pred_mean[:, i]

        pred_mean = self._unweight_y(torch.tensor(pred_mean), opt_metric_settings).numpy()
        pred_std = self._unweight_y(torch.tensor(pred_std), opt_metric_settings).numpy()

        return best_samples, recommend_type, pred_mean, pred_std

    def _get_exploit_or_explore(self) -> List[str]:
        with torch.no_grad():
            posterior = self.global_model.posterior(self.acq_result)
            pred_mean = posterior.mean

        hvi_values = torch.tensor([compute_hvi(pred_mean[i], self.pareto_y, self.ref_point) for i in range(pred_mean.shape[0])])
        ehvi_values = self.acq_value.to(device="cpu")
        exploit_scores = hvi_values / (ehvi_values + 1e-6)
        explore_scores = 1 - exploit_scores

        for i in range(self.acq_result.shape[0]):
            self.console.log(
                f"Point {i}: "
                f"EHVI = {ehvi_values[i]:.3f}, "
                f"HVI = {hvi_values[i]:.3f}, "
                f"Exploit Score = {exploit_scores[i]:.3f}, "
                f"Explore Score = {explore_scores[i]:.3f}"
            )
        return ["exploit" if exploit_scores[i] > explore_scores[i] else "explore" for i in range(self.acq_result.shape[0])]

    def _get_predictions(self) -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            posterior = self.global_model.posterior(self.acq_result)
            pred_mean = posterior.mean.cpu().numpy()
            pred_var = posterior.variance.cpu().numpy()
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
