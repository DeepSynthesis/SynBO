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
    ): ...
