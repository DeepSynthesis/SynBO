import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from rich.console import Console
import math
from copy import deepcopy


class DesignSpace:
    """Simplified design space implementation"""

    def __init__(self):
        self.paras = {}
        self.para_names = []
        self.num_paras = 0
        self.num_numeric = 0
        self.num_categorical = 0
        self.opt_lb = None
        self.opt_ub = None

    def parse(self, para_config: Dict):
        """Parse parameter configuration"""
        self.para_names = list(para_config.keys())
        self.num_paras = len(self.para_names)
        self.num_numeric = 0
        self.num_categorical = 0

        # For simplicity, we'll treat all as numeric in this implementation
        self.num_numeric = self.num_paras
        self.num_categorical = 0

        # Set bounds
        self.opt_lb = np.array([para_config[name][1][0] for name in self.para_names])
        self.opt_ub = np.array([para_config[name][1][1] for name in self.para_names])

        return self

    def transform(self, x: pd.DataFrame):
        """Transform dataframe to arrays"""
        if isinstance(x, pd.DataFrame):
            x_numeric = x[self.para_names].values
            x_enum = None
        else:
            x_numeric = x
            x_enum = None
        return x_numeric, x_enum

    def inverse_transform(self, x, xe=None):
        """Inverse transform arrays to dataframe"""
        df = pd.DataFrame(x, columns=self.para_names)
        return df


class SurrogateModel:
    """Simple surrogate model using Gaussian Process approximation"""

    def __init__(self, num_dims: int):
        self.num_dims = num_dims
        self.X_train = None
        self.y_train = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model"""
        self.X_train = X.copy()
        self.y_train = y.copy().reshape(-1, 1)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mean and variance"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        # Simple nearest neighbor prediction for demonstration
        # In a real implementation, this would be a proper GP
        means = []
        variances = []

        for x in X:
            # Find nearest neighbors
            distances = np.linalg.norm(self.X_train - x, axis=1)
            nearest_idx = np.argmin(distances)

            # Return the nearest neighbor's value with some uncertainty
            mean = self.y_train[nearest_idx, 0]
            variance = np.exp(-distances[nearest_idx])  # Decreases with proximity

            means.append(mean)
            variances.append(variance)

        return np.array(means), np.array(variances)


class MACEAcquisition:
    """Simplified MACE acquisition function"""

    def __init__(self, model, best_y, kappa=2.0):
        self.model = model
        self.best_y = best_y
        self.kappa = kappa

    def __call__(self, X):
        """Evaluate acquisition function"""
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()

        mean, variance = self.model.predict(X)
        std = np.sqrt(variance)

        # Lower confidence bound
        lcb = mean - self.kappa * std

        # MACE combines improvement with exploration
        improvement = np.maximum(self.best_y - lcb, 0)
        acquisition = improvement + std  # Balance exploitation and exploration

        return torch.tensor(acquisition)


class EvolutionOptimizer:
    """Simplified evolution optimizer"""

    def __init__(self, space, acq_func, pop=50, iters=50, verbose=False):
        self.space = space
        self.acq_func = acq_func
        self.pop = pop
        self.iters = iters
        self.verbose = verbose

    def optimize(self, initial_suggest=None, fix_input=None):
        """Optimize using evolutionary algorithm"""
        # Simplified implementation using random search with local optimization
        best_candidates = []

        # Generate initial population
        pop_X = np.random.uniform(self.space.opt_lb, self.space.opt_ub, size=(self.pop, self.space.num_paras))

        # Evaluate population
        pop_acq = self.acq_func(torch.tensor(pop_X)).detach().cpu().numpy()

        # Evolutionary loop
        for _ in range(self.iters):
            # Select best individuals
            elite_idx = np.argsort(pop_acq)[-self.pop // 2 :]
            elite_X = pop_X[elite_idx]

            # Generate new population through mutation and crossover
            new_pop = []
            for _ in range(self.pop):
                # Select parents
                parent1 = elite_X[np.random.randint(len(elite_X))]
                parent2 = elite_X[np.random.randint(len(elite_X))]

                # Crossover
                crossover_point = np.random.randint(1, self.space.num_paras)
                child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])

                # Mutation
                mutation_rate = 0.1
                mutation_strength = 0.1
                for i in range(self.space.num_paras):
                    if np.random.random() < mutation_rate:
                        child[i] += np.random.normal(0, mutation_strength * (self.space.opt_ub[i] - self.space.opt_lb[i]))

                # Bounds enforcement
                child = np.clip(child, self.space.opt_lb, self.space.opt_ub)
                new_pop.append(child)

            pop_X = np.array(new_pop)
            pop_acq = self.acq_func(torch.tensor(pop_X)).detach().cpu().numpy()

        # Return best candidates
        best_idx = np.argsort(pop_acq)[-10:]  # Return top 10
        best_X = pop_X[best_idx]

        # Convert to DataFrame
        df_result = pd.DataFrame(best_X, columns=self.space.para_names)

        # Apply fixed inputs if provided
        if fix_input is not None:
            for k, v in fix_input.items():
                df_result[k] = v

        return df_result


class HEBOOptimizer:
    """Simplified HEBO optimizer implementation"""

    def __init__(self, space, rand_sample=10):
        self.space = space
        self.X = pd.DataFrame(columns=self.space.para_names)
        self.y = np.zeros((0, 1))
        self.rand_sample = rand_sample

    def suggest(self, n_suggestions=1, fix_input=None):
        """Suggest next points to evaluate"""
        # Skip random sampling phase entirely since all data is provided upfront
        if self.X.shape[0] == 0:
            # No data provided, return random samples
            samp = np.random.uniform(self.space.opt_lb, self.space.opt_ub, size=(n_suggestions, self.space.num_paras))
            df_samp = pd.DataFrame(samp, columns=self.space.para_names)

            if fix_input is not None:
                for k, v in fix_input.items():
                    df_samp[k] = v

            return df_samp
        else:
            # Model-based optimization phase
            # Fit surrogate model
            X_trans, _ = self.space.transform(self.X)
            model = SurrogateModel(self.space.num_paras)
            model.fit(X_trans, self.y.flatten())

            # Get best value
            best_y = self.y.min()

            # Create acquisition function
            acq = MACEAcquisition(model, best_y=best_y, kappa=2.0)

            # Optimize acquisition function
            opt = EvolutionOptimizer(self.space, acq, pop=50, iters=30, verbose=False)
            rec = opt.optimize(fix_input=fix_input)

            # Remove duplicates and ensure we have enough suggestions
            rec = rec.drop_duplicates()

            # If we don't have enough unique suggestions, add random ones
            if len(rec) < n_suggestions:
                additional_samp = np.random.uniform(
                    self.space.opt_lb, self.space.opt_ub, size=(n_suggestions - len(rec), self.space.num_paras)
                )
                df_additional = pd.DataFrame(additional_samp, columns=self.space.para_names)
                rec = pd.concat([rec, df_additional], ignore_index=True)

            # Return requested number of suggestions
            if len(rec) >= n_suggestions:
                selected_idx = np.random.choice(len(rec), n_suggestions, replace=False)
                return rec.iloc[selected_idx]
            else:
                return rec

    def observe(self, X, y):
        """Observe new data points"""
        valid_id = np.where(np.isfinite(y.reshape(-1)))[0].tolist()
        XX = X.iloc[valid_id] if isinstance(X, pd.DataFrame) else pd.DataFrame(X[valid_id], columns=self.space.para_names)
        yy = y[valid_id].reshape(-1, 1)
        self.X = pd.concat([self.X, XX], axis=0, ignore_index=True)
        self.y = np.vstack([self.y, yy])


class HEBOOptimizerWrapper:
    """Wrapper class for HEBO optimization in reactionopt framework"""

    def __init__(self, name_data: np.ndarray, mc_num_samples: int = 64, max_batch_size: int = 128, seed: int = 1145141):
        """
        Initialize HEBO optimizer wrapper

        Args:
            name_data: Array of condition names
            mc_num_samples: Number of Monte Carlo samples (not used in this simplified version)
            max_batch_size: Maximum batch size for optimization
            seed: Random seed
        """
        self.name_data = name_data
        self.mc_num_samples = mc_num_samples
        self.seed = seed
        self.max_batch_size = max_batch_size
        self.console = Console()

        # Don't set fixed random seed to allow different results each time
        # np.random.seed(seed)

    def optimize(
        self,
        training_X: np.ndarray,
        training_y: Dict[str, np.ndarray],
        candidate_X: np.ndarray,
        opt_direct_info: List[Dict],
        device: torch.device,
        batch_size: int = 5,
        opt_weights: Dict = None,
        maximum_metrics: bool = True,
    ) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:
        """
        Optimize using enhanced HEBO algorithm

        Args:
            training_X: Normalized training inputs
            training_y: Normalized training outputs (dictionary)
            candidate_X: All possible candidate points
            opt_direct_info: Optimization direction info
            device: PyTorch device
            batch_size: Number of points to select
            opt_weights: Optional weights for multi-objective optimization
            maximum_metrics: Whether to maximize metrics

        Returns:
            Tuple of (selected_conditions, recommend_type, pred_mean, pred_std)
        """
        # Convert training_y dict to array
        if isinstance(training_y, dict):
            training_y_array = np.array(list(training_y.values())).T
        else:
            training_y_array = training_y

        # Determine number of objectives
        num_objectives = training_y_array.shape[1] if len(training_y_array.shape) > 1 else 1

        # Create design space for HEBO
        space_config = {}
        for i in range(training_X.shape[1]):
            # Assuming normalized space [-1, 1] - you may need to adjust based on your data
            space_config[f"x{i}"] = ("num", [-1.0, 1.0])

        space = DesignSpace().parse(space_config)

        # Create enhanced HEBO optimizer with better exploration
        hebo_optimizer = HEBOOptimizer(space=space, rand_sample=0)

        # Observe training data
        df_X = pd.DataFrame(training_X, columns=[f"x{i}" for i in range(training_X.shape[1])])
        # For multi-objective, we'll use a simple aggregation (you might want to improve this)
        if num_objectives > 1:
            # Simple weighted sum (or use first objective if no weights provided)
            if opt_weights is not None and len(opt_weights) == num_objectives:
                aggregated_y = np.dot(training_y_array, opt_weights)
            else:
                aggregated_y = training_y_array[:, 0]  # Use first objective
        else:
            aggregated_y = training_y_array.flatten()

        hebo_optimizer.observe(df_X, aggregated_y)

        # Get suggestions from enhanced HEBO
        suggestions_df = hebo_optimizer.suggest(n_suggestions=batch_size)

        # Convert suggestions to numpy array
        suggestions_np = suggestions_df.values

        # Find closest candidates in the candidate set with diversity enhancement
        selected_indices = []
        best_samples = []

        # Keep track of selected points to ensure diversity
        selected_points = []

        for i, suggestion in enumerate(suggestions_np):
            # Find the closest candidate point
            distances = np.linalg.norm(candidate_X - suggestion, axis=1)

            # Penalize distances to already selected points to encourage diversity
            if selected_points:
                selected_array = np.array(selected_points)
                diversity_penalty = np.min(np.linalg.norm(selected_array - suggestion, axis=1))
                # Modify distances to favor diverse selections
                distances = distances - 0.1 * diversity_penalty  # Encourage diversity

            closest_idx = np.argmin(distances)
            selected_indices.append(closest_idx)
            best_samples.append(candidate_X[closest_idx])
            selected_points.append(candidate_X[closest_idx])

        selected_indices = np.array(selected_indices)
        best_samples = np.array(best_samples)

        # Get selected conditions
        selected_conditions = self.name_data[selected_indices].squeeze()

        # Enhanced recommendation type determination
        # In a more complete implementation, you could extract these from the HEBO model
        recommend_type = []
        for i in range(batch_size):
            # Simple heuristic: alternate between exploit and explore,
            # or use more explore in early stages
            if len(aggregated_y) < 10 and i < batch_size // 2:
                recommend_type.append("explore")
            elif i % 2 == 0:
                recommend_type.append("exploit")
            else:
                recommend_type.append("explore")

        # For prediction values, we'll return dummy values
        # In a more complete implementation, you could extract these from the HEBO model
        pred_mean = np.zeros((batch_size, num_objectives))
        pred_std = np.ones((batch_size, num_objectives))

        self.console.print("✅ Finish optimization with enhanced HEBO", style="green")
        return selected_conditions, recommend_type, pred_mean, pred_std
