from typing import List
import numpy as np
import torch
from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn, TimeRemainingColumn

from rxnopt.utils.logger import console
from rxnopt.bo_algorithm.bo_core import DefaultBO


class Optimizer:
    def __init__(self, method, name_data, mc_num_samples: int = 64, max_batch_size: int = 128, random_seed: int = 42):
        self.method = method
        self.name_data = name_data
        self.opt_console = console

        if self.method == "BO":
            self.bo_optimizer = DefaultBO(mc_num_samples=mc_num_samples, max_batch_size=max_batch_size, random_seed=random_seed)

    def optimize(
        self,
        training_X: np.ndarray,
        training_y: np.ndarray,
        candidate_X: np.ndarray,
        opt_metric_setting: List[dict],
        device: torch.device,
        batch_size: int = 5,
    ) -> List[int]:
        for k, d in zip(training_y.keys(), opt_metric_setting):
            if d["opt_direct"] == "min":
                training_y[k] = -training_y[k]

        training_y_dict = training_y.copy()
        if isinstance(training_y, dict):
            training_y = np.array(list(training_y.values())).T

        if self.method == "newBO":
            raise Exception("newBO method has not been implemented yet.")
        elif self.method == "random":
            raise Exception("random method has not been implemented yet.")
        elif self.method == "BO":
            training_X_t = torch.tensor(training_X).double()
            training_y_t = torch.tensor(training_y).double()
            training_y_t = self._weight_y(training_y_t, opt_metric_setting)
            candidate_X_t = torch.tensor(candidate_X).double()
            dtype = torch.double
            training_X_t = training_X_t.to(device=device, dtype=dtype)
            training_y_t = training_y_t.to(device=device, dtype=dtype)
            candidate_X_t = candidate_X_t.to(device=device, dtype=dtype)

            with Progress(
                TextColumn("[bold cyan]{task.description}"),
                BarColumn(bar_width=None),
                MofNCompleteColumn(),
                TimeRemainingColumn(),
                console=self.opt_console,
            ) as progress:
                num_models = training_y_t.shape[1]
                task_train = progress.add_task(description="Training surrogate models", total=num_models, start=True)
                task_pareto = progress.add_task(description="Calculating Pareto frontiers", total=len(training_y) - 1)
                task_acq_opt = progress.add_task(description="Optimizing acquisition function", total=batch_size)

                acq_result, acq_value = self.bo_optimizer.optimize(
                    training_X_t=training_X_t,
                    training_y_t=training_y_t,
                    candidate_X_t=candidate_X_t,
                    opt_metric_setting=opt_metric_setting,
                    device=device,
                    batch_size=batch_size,
                    progress=progress,
                    task_train=task_train,
                    task_pareto=task_pareto,
                    task_acq_opt=task_acq_opt,
                    training_y_dict=training_y_dict,
                    opt_console=self.opt_console,
                )

            if device.type == "cuda":
                best_samples = [res.cpu().numpy() for res in acq_result]
            else:
                best_samples = [res.numpy() for res in acq_result]

            recommend_type = self.bo_optimizer.get_exploit_or_explore(acq_value, self.opt_console)

            selected_indices = [np.argwhere(np.all(candidate_X == best_sample, axis=1)).flatten() for best_sample in best_samples]
            selected_indices = np.array(selected_indices).squeeze()
            selected_conditions = self.name_data[selected_indices].squeeze()

            pred_mean, pred_std = self.bo_optimizer.get_predictions()

            for i, d in enumerate(opt_metric_setting):
                if d["opt_direct"] == "min":
                    pred_mean[:, i] = -pred_mean[:, i]

            pred_mean = self._unweight_y(torch.tensor(pred_mean), opt_metric_setting).numpy()
            pred_std = self._unweight_y(torch.tensor(pred_std), opt_metric_setting).numpy()

            self.opt_console.print("✅ Finish optimization", style="green")
            return selected_conditions, recommend_type, pred_mean, pred_std
        else:
            raise Exception(f"Unknown optimization method: {self.method}")

    def _weight_y(self, training_y: torch.Tensor, opt_metric_setting: List[dict]) -> torch.Tensor:
        weights = torch.tensor([d["metric_weight"] for d in opt_metric_setting])

        training_y = training_y * weights
        return training_y

    def _unweight_y(self, training_y: torch.Tensor, opt_metric_setting: List[dict]) -> torch.Tensor:
        weights = torch.tensor([d["metric_weight"] for d in opt_metric_setting])
        training_y = training_y / weights
        return training_y
