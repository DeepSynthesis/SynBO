from typing import List
import numpy as np
import torch

from synbo.algorithm.evolution import DefaultEO
from synbo.algorithm.particle_swarm import DefaultPS
from synbo.utils.logger import console
from synbo.algorithm.bo_core import DefaultBO
from synbo.algorithm.random_select import RandomSelect


class Optimizer:
    def __init__(
        self,
        method,
        random_seed: int = 42,
        device: torch.device = torch.device("cpu"),
        optimization_kwargs: dict = None,
    ):
        self.method = method
        self.random_seed = random_seed
        self.opt_console = console

        if self.method == "default_BO":
            self.optimizer = DefaultBO(random_seed=random_seed, device=device, **optimization_kwargs)
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
        done_name: np.ndarray = None,
        candidate_name: np.ndarray = None,
        condition_types: List[str] = None,
    ) -> List[int]:
        for i, d in enumerate(opt_metric_settings):
            if d["opt_direct"] == "min":
                training_y[i] = -training_y[i]

        training_y = training_y.T

        if self.method in ["default_BO"]:  # , "random_select", "evolution", "particle_swarm"]:
            best_samples, recommend_type, pred_mean, pred_std = self.optimizer.optimize(
                training_X=training_X,
                training_y=training_y,
                candidate_X=candidate_X,
                opt_metric_settings=opt_metric_settings,
                batch_size=batch_size,
                done_name=done_name,
                candidate_name=candidate_name,
                condition_types=condition_types,
            )

            from IPython import embed; embed(); exit()
            
            selected_indices = [np.argwhere(np.all(candidate_X == best_sample, axis=1)).flatten() for best_sample in best_samples]
            # from IPython import embed; embed()
            selected_indices = np.array(selected_indices).squeeze()
            selected_conditions = candidate_name[selected_indices].squeeze()

            self.opt_console.print("✅ Finish optimization", style="green")
            return selected_conditions, recommend_type, pred_mean, pred_std
        else:
            raise Exception(f"Unknown optimization method: {self.method}")
