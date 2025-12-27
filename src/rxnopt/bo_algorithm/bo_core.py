from typing import List
import numpy as np
import torch
from rich.progress import Progress

from botorch.models import ModelListGP
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.sampling.normal import SobolQMCNormalSampler

from rxnopt.utils.util_func import compute_hvi
from rxnopt.bo_algorithm.GP_opt import GPSurrogateModel
from rxnopt.bo_algorithm.acf_opt import optimize_acqf_discrete, EHVIAcquisitionFunction, ParetoFrontCalculator

import warnings
from linear_operator.utils.cholesky import NumericalWarning

warnings.filterwarnings("ignore", category=NumericalWarning)


class DefaultBO:
    def __init__(self, mc_num_samples: int = 64, max_batch_size: int = 128, random_seed: int = 42):
        self.mc_num_samples = mc_num_samples
        self.random_seed = random_seed
        self.max_batch_size = max_batch_size
        self.surrogate_model_class = GPSurrogateModel
        self.acquisition_function_class = EHVIAcquisitionFunction
        self.target_evaluator = ParetoFrontCalculator()

    def optimize(
        self,
        training_X_t: torch.Tensor,
        training_y_t: torch.Tensor,
        candidate_X_t: torch.Tensor,
        opt_metric_setting: List[dict],
        device: torch.device,
        batch_size: int,
        progress: Progress,
        task_train,
        task_pareto,
        task_acq_opt,
        training_y_dict: dict,
        opt_console,
    ):
        models = []
        num_models = training_y_t.shape[1]
        for i in range(num_models):
            key = list(training_y_dict.keys())[i]
            progress.log(f"Fitting model for [bold]{key}[/bold]...", style="yellow")

            train_y_i = training_y_t[:, i].reshape(-1, 1)
            model_i = self.surrogate_model_class(device=device, num_dims=training_X_t.shape[1])
            model_i.fit(training_X_t, train_y_i)
            models.append(model_i.model)
            progress.update(task_train, advance=1)

        self.global_model = ModelListGP(*models)

        training_y_np = training_y_t.cpu().numpy()
        self.pareto_y = self.target_evaluator.calculate_target_function(training_y_np, progress, task_pareto).to(device=device)

        self.ref_point = torch.tensor([-1 if omi["opt_direct"] == "min" else 0 for omi in opt_metric_setting], dtype=float, device=device)

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.mc_num_samples]), seed=self.random_seed)
        partitioning = NondominatedPartitioning(ref_point=self.ref_point, Y=self.pareto_y)
        acq_func = self.acquisition_function_class(
            model=self.global_model,
            sampler=sampler,
            ref_point=self.ref_point,
            partitioning=partitioning,
        )

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

        return self.acq_result, self.acq_value

    def get_exploit_or_explore(self, acq_value, opt_console):
        with torch.no_grad():
            posterior = self.global_model.posterior(self.acq_result)
            pred_mean = posterior.mean
            pred_var = posterior.variance

        hvi_values = torch.tensor([compute_hvi(pred_mean[i], self.pareto_y, self.ref_point) for i in range(pred_mean.shape[0])])
        ehvi_values = acq_value.to(device="cpu")
        exploit_scores = hvi_values / (ehvi_values + 1e-6)
        explore_scores = 1 - exploit_scores

        for i in range(self.acq_result.shape[0]):
            opt_console.log(
                f"Point {i}: "
                f"EHVI = {ehvi_values[i]:.3f}, "
                f"HVI = {hvi_values[i]:.3f}, "
                f"Exploit Score = {exploit_scores[i]:.3f}, "
                f"Explore Score = {explore_scores[i]:.3f}"
            )
        return ["exploit" if exploit_scores[i] > explore_scores[i] else "explore" for i in range(self.acq_result.shape[0])]

    def get_predictions(self):
        with torch.no_grad():
            posterior = self.global_model.posterior(self.acq_result)
            pred_mean = posterior.mean.cpu().numpy()
            pred_var = posterior.variance.cpu().numpy()
            pred_std = np.sqrt(pred_var)
        return pred_mean, pred_std
