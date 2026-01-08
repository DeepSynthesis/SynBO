from typing import List
import numpy as np
import torch

from rxnopt.algorithm.evolution import DefaultEO
from rxnopt.utils.logger import console
from rxnopt.algorithm.bo_core import DefaultBO
from rxnopt.algorithm.random_select import RandomSelect


class Optimizer:
    def __init__(
        self, method, name_data, random_seed: int = 42, device: torch.device = torch.device("cpu"), optimization_kwargs: dict = None
    ):
        self.method = method
        self.name_data = name_data
        self.random_seed = random_seed
        self.opt_console = console

        if self.method == "default_BO":
            self.optimizer = DefaultBO(random_seed=random_seed, device=device, **optimization_kwargs)
        # elif self.method == "HEBO":
        #     self.optimizer = HEBOOptimizer(random_seed=random_seed, sedp=sedp, ffm=ffm)
        elif self.method == "evolution":
            self.optimizer = DefaultEO(random_seed=random_seed, device=device, **optimization_kwargs)
        elif self.method == "particle_swarm":
            self.optimizer = DefaultPS(random_seed=random_seed, device=device, **optimization_kwargs)
        elif self.method == "random_select":
            self.optimizer = RandomSelect(random_seed=random_seed)

        else:
            raise Exception(f"Unknown optimization method: {self.method}")

    def optimize(
        self,
        training_X: np.ndarray,
        training_y: np.ndarray,
        candidate_X: np.ndarray,
        opt_metric_settings: List[dict],
        batch_size: int = 5,
    ) -> List[int]:
        for k, d in zip(training_y.keys(), opt_metric_settings):
            if d["opt_direct"] == "min":
                training_y[k] = -training_y[k]

        training_y_dict = training_y.copy()
        if isinstance(training_y, dict):
            training_y = np.array(list(training_y.values())).T

        if self.method in ["default_BO", "random_select", "evolution"]:
            best_samples, recommend_type, pred_mean, pred_std = self.optimizer.optimize(
                training_X=training_X,
                training_y=training_y,
                candidate_X=candidate_X,
                opt_metric_settings=opt_metric_settings,
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
