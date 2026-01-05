from typing import List
import numpy as np


class RandomSelect:
    def __init__(self, random_seed: int = 42):

        self.random_seed = random_seed
        np.random.seed(self.random_seed)

    def optimize(
        self,
        training_X: np.ndarray,
        training_y: np.ndarray,
        candidate_X: np.ndarray,
        opt_metric_settings: List[dict],
        batch_size: int,
        training_y_dict: dict,
    ):
        training_set = {tuple(row) for row in training_X}

        available_indices = [i for i, row in enumerate(candidate_X) if tuple(row) not in training_set]
        num_available = len(available_indices)
        if num_available < batch_size:
            raise ValueError(
                f"Insufficient number of samplable samples: {batch_size} are needed, but after excluding existing points, only {num_available} remain."
            )
        new_candidate_indices = np.random.choice(available_indices, size=batch_size, replace=False)
        new_candidate_samples = candidate_X[new_candidate_indices]

        recommend_type = ["random_select"] * batch_size
        return (
            new_candidate_samples,
            recommend_type,
            np.array([[0.0] * len(opt_metric_settings)] * batch_size),
            np.array([[0.0] * len(opt_metric_settings)] * batch_size),
        )
