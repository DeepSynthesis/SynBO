from typing import List
import numpy as np
import torch

from rxnopt.utils.logger import console
from rxnopt.bo_algorithm.bo_core import DefaultBO


class Optimizer:
    def __init__(self, method, name_data, mc_num_samples: int = 64, max_batch_size: int = 128, random_seed: int = 42):
        self.method = method
        self.name_data = name_data
        self.opt_console = console

        if self.method == "default_BO":
            self.optimizer = DefaultBO(mc_num_samples=mc_num_samples, max_batch_size=max_batch_size, random_seed=random_seed, opt_console=console)
        else:
            raise Exception(f"Unknown optimization method: {self.method}")

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

        if self.method == "default_BO":
            best_samples, recommend_type, pred_mean, pred_std = self.optimizer.optimize(
                training_X=training_X,
                training_y=training_y,
                candidate_X=candidate_X,
                opt_metric_setting=opt_metric_setting,
                device=device,
                batch_size=batch_size,
                training_y_dict=training_y_dict,
            )

            selected_indices = [np.argwhere(np.all(candidate_X == best_sample, axis=1)).flatten() for best_sample in best_samples]
            selected_indices = np.array(selected_indices).squeeze()
            selected_conditions = self.name_data[selected_indices].squeeze()

            self.opt_console.print("✅ Finish optimization", style="green")
            return selected_conditions, recommend_type, pred_mean, pred_std
        else:
            raise Exception(f"Unknown optimization method: {self.method}")
