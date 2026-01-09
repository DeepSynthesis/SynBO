from typing import List, Tuple
import numpy as np
import torch
from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn, TimeRemainingColumn

from botorch.models import ModelListGP
from rxnopt.utils.logger import console
from rxnopt.algorithm.sg_model import GPSurrogateModel, SklearnModelWrapper

import warnings
from linear_operator.utils.cholesky import NumericalWarning

warnings.filterwarnings("ignore", category=NumericalWarning)


class DefaultPS:
    def __init__(
        self,
        random_seed: int = 42,
        surrogate_model: str = "GP",
        device: torch.device = torch.device("cpu"),
        accuracy: str = "medium",
        # PSO specific parameters
        w: float = 0.729,  # Inertia weight
        c1: float = 1.494,  # Cognitive weight
        c2: float = 1.494,  # Social weight
    ):
        self.random_seed = random_seed
        self.console = console
        self.device = device

        # Hyperparameters for PSO
        self.w = w
        self.c1 = c1
        self.c2 = c2

        if accuracy == "medium":
            self.num_particles = 100
            self.max_iter = 50

        if surrogate_model == "GP":
            self.surrogate_model_class = GPSurrogateModel
        elif surrogate_model == "RF":
            from rxnopt.algorithm.sg_model import RFSurrogateModel
            self.surrogate_model_class = RFSurrogateModel
        else:
            raise ValueError(f"Unknown surrogate model: {surrogate_model}")

    def optimize(
        self,
        training_X: np.ndarray,
        training_y: np.ndarray,
        candidate_X: np.ndarray,
        opt_metric_settings: List[dict],
        batch_size: int,
        training_y_dict: dict,
    ) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:

        # 1. Data deduplication
        existing_set = set(map(tuple, training_X))
        keep_mask = np.array([tuple(x) not in existing_set for x in candidate_X])

        n_original = len(candidate_X)
        candidate_X = candidate_X[keep_mask]
        n_filtered = len(candidate_X)

        if n_original != n_filtered:
            self.console.log(f"Removed {n_original - n_filtered} duplicates from candidate set.", style="dim")

        if n_filtered == 0:
            warnings.warn("All candidates appear in training data. Returning empty results.")
            return np.array([]), [], np.array([]), np.array([])

        current_batch_size = min(batch_size, n_filtered)

        training_X_t = torch.tensor(training_X).double().to(device=self.device)
        candidate_X_t = torch.tensor(candidate_X).double().to(device=self.device)

        # Process Y:
        # For PSO (Maximization), we flip signs for 'min' objectives so we can maximize everything internally.
        # We also apply weights here for the surrogate model to learn the weighted scale if desired,
        # but typically we train on raw/normalized data and weight the fitness.
        # Here we follow the BO pattern: apply weights early.
        training_y_t = torch.tensor(training_y).double()
        training_y_t = self._pre_process_y(training_y_t, opt_metric_settings).to(device=self.device)

        # 2. Train Surrogate Models (Identical to BO)
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
                model_i = self.surrogate_model_class(device=self.device, num_dims=training_X_t.shape[1])

                if isinstance(model_i, GPSurrogateModel):
                    model_i.fit(training_X_t, train_y_i)
                    models.append(model_i.model)
                else:
                    wrapper = SklearnModelWrapper(model_i)
                    wrapper.fit_surrogate(training_X_t, train_y_i)
                    models.append(wrapper)

                progress.update(task_train, advance=1)

            self.global_model = ModelListGP(*models)

            # 3. Particle Swarm Optimization Loop
            task_pso = progress.add_task(description="Running PSO optimization", total=self.max_iter)

            # Define bounds based on candidate_X or provided range
            # Assuming candidate_X covers the feasible space
            lower_bounds = candidate_X_t.min(dim=0)[0]
            upper_bounds = candidate_X_t.max(dim=0)[0]
            dim = training_X_t.shape[1]

            # Initialize Particles
            # Random initialization within bounds
            particles = lower_bounds + (upper_bounds - lower_bounds) * torch.rand(
                self.num_particles, dim, device=self.device, dtype=torch.double
            )
            velocities = torch.zeros_like(particles)

            pbest_pos = particles.clone()
            pbest_scores = torch.full((self.num_particles,), -float("inf"), device=self.device, dtype=torch.double)

            # Global best (track top batch_size candidates found across history)
            # For simplicity in standard PSO, we track the single scalar best,
            # but we will store a history of good points to select from later.
            gbest_pos = particles[0].clone()
            gbest_score = -float("inf")

            # To avoid selecting the same point multiple times, we collect good candidates
            candidate_pool_pos = []
            candidate_pool_scores = []

            for iteration in range(self.max_iter):
                # Evaluate fitness using Surrogate Model
                fitness_scores = self._evaluate_fitness(particles)

                # Update Personal Bests
                improved_indices = fitness_scores > pbest_scores
                pbest_pos[improved_indices] = particles[improved_indices]
                pbest_scores[improved_indices] = fitness_scores[improved_indices]

                # Update Global Best
                max_score, max_idx = torch.max(fitness_scores, dim=0)
                if max_score > gbest_score:
                    gbest_score = max_score
                    gbest_pos = particles[max_idx].clone()

                # Store current bests for final selection
                candidate_pool_pos.append(particles.clone())
                candidate_pool_scores.append(fitness_scores.clone())

                # Update Velocities and Positions
                r1 = torch.rand_like(particles)
                r2 = torch.rand_like(particles)

                velocities = self.w * velocities + self.c1 * r1 * (pbest_pos - particles) + self.c2 * r2 * (gbest_pos - particles)
                particles = particles + velocities

                # Clamp boundaries
                particles = torch.max(torch.min(particles, upper_bounds), lower_bounds)

                progress.update(task_pso, advance=1)

        # 4. Selection of Best Samples
        # Concatenate all history
        all_candidates = torch.cat(candidate_pool_pos, dim=0)
        all_scores = torch.cat(candidate_pool_scores, dim=0)

        # Sort by score descending
        sorted_indices = torch.argsort(all_scores, descending=True)
        top_continuous_X = all_candidates[sorted_indices]

        # Map continuous PSO results to nearest neighbors in candidate_X
        # We need `batch_size` unique points.
        best_indices_in_candidate = []

        # Efficient distance calculation using batches if needed, here doing simple loop for clarity
        # We take top 3*batch_size continuous points to map, to ensure we find enough unique discrete points
        check_limit = min(len(top_continuous_X), batch_size * 5)

        # Calculate pairwise distances between Top PSO points and Discrete Candidates
        # shape: (check_limit, num_candidates)
        dists = torch.cdist(top_continuous_X[:check_limit], candidate_X_t)

        # Find nearest indices
        nearest_indices = torch.argmin(dists, dim=1)

        seen_idx = set()
        # Also exclude training points if possible (optional, based on logic)
        # Here we just ensure uniqueness in the batch

        final_indices = []
        for idx in nearest_indices.tolist():
            if idx not in seen_idx:
                final_indices.append(idx)
                seen_idx.add(idx)
            if len(final_indices) >= batch_size:
                break

        # If we didn't find enough unique points (rare), fill with random top ones
        if len(final_indices) < batch_size:
            remaining = [i for i in range(len(candidate_X)) if i not in seen_idx]
            final_indices.extend(remaining[: batch_size - len(final_indices)])

        best_samples_idx = final_indices
        best_samples = candidate_X[best_samples_idx]

        # 5. Get predictions for the selected samples
        best_samples_t = candidate_X_t[best_samples_idx]
        pred_mean, pred_std = self._get_predictions(best_samples_t)

        # Remove weights to return to original scale
        pred_mean = self._unweight_y(torch.tensor(pred_mean), opt_metric_settings).numpy()
        pred_std = self._unweight_y(torch.tensor(pred_std), opt_metric_settings).numpy()

        recommend_type = ["pso_exploit"] * batch_size

        return best_samples, recommend_type, pred_mean, pred_std

    def _evaluate_fitness(self, X: torch.Tensor) -> torch.Tensor:
        """
        Evaluate particles using the Surrogate Model.
        Fitness = Weighted Sum of Means + Beta * Std (UCB-like scalarization)
        """
        with torch.no_grad():
            posterior = self.global_model.posterior(X)
            mean = posterior.mean  # (num_particles, num_objectives)
            # We can include variance for exploration if desired,
            # here we stick to mean (exploitation) which is standard for PSO,
            # or add a small UCB term.
            # var = posterior.variance

            # Since data is already weighted and signs flipped in _pre_process_y,
            # we simply sum the objectives to get a scalar fitness.
            scalar_fitness = mean.sum(dim=1)

        return scalar_fitness

    def _get_predictions(self, X: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            posterior = self.global_model.posterior(X)
            pred_mean = posterior.mean.cpu().numpy()
            pred_var = posterior.variance.cpu().numpy()
            pred_std = np.sqrt(pred_var)
        return pred_mean, pred_std

    def _pre_process_y(self, training_y: torch.Tensor, opt_metric_settings: List[dict]) -> torch.Tensor:
        """
        Applies weights and flips signs so that the internal optimizer
        always performs Maximization on Weighted Data.
        """
        processed_y = training_y.clone()
        weights = torch.tensor([d["metric_weight"] for d in opt_metric_settings], device=training_y.device, dtype=torch.double)

        # Apply weights first
        processed_y = processed_y * weights

        # Flip signs for minimization tasks
        for i, setting in enumerate(opt_metric_settings):
            if setting["opt_direct"] == "min":
                processed_y[:, i] = -processed_y[:, i]

        return processed_y

    def _unweight_y(self, training_y: torch.Tensor, opt_metric_settings: List[dict]) -> torch.Tensor:
        weights = torch.tensor([d["metric_weight"] for d in opt_metric_settings], dtype=torch.double)
        training_y = training_y / weights
        return training_y
