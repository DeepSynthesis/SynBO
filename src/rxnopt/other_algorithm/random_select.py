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
        opt_metric_setting: List[dict],
        device: torch.device,
        batch_size: int,
        training_y_dict: dict,
    ) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:

        num_candidates = candidate_X.shape[0]
        selected_indices = np.random.choice(num_candidates, size=batch_size, replace=False)
        best_samples = candidate_X[selected_indices]

        recommend_type = "random_selection"
        pred_mean = None
        pred_std = None

        return best_samples, recommend_type, pred_mean, pred_std
