from typing import List, Tuple
import numpy as np
import torch
from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn, TimeRemainingColumn

from botorch.models import ModelListGP
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from botorch.sampling.normal import SobolQMCNormalSampler

from rxnopt.utils.util_func import compute_hvi, generate_constraint_mask
from rxnopt.utils.logger import console
from rxnopt.algorithm.sg_model import (
    BNNEnsembleSurrogateModel,
    BayesianLinearSurrogateModel,
    GPSurrogateModel,
    RFSurrogateModel,
    SklearnModelWrapper,
)
from rxnopt.algorithm.acq_function import (
    EHVIAcquisitionFunction,
    NEIAcquisitionFunction,
    ParEGOAcquisitionFunction,
    UCBAcquisitionFunction,
    ParetoFrontCalculator,
)

import warnings
from linear_operator.utils.cholesky import NumericalWarning

warnings.filterwarnings("ignore", category=NumericalWarning)


class DefaultBO:
    def __init__(
        self,
        random_seed: int = 42,
        surrogate_model: str = "GP",
        acq_func: str = "EHVI",
        device: torch.device = torch.device("cpu"),
        accuracy: str = "medium",
    ):
        self.random_seed = random_seed
        self.console = console

        if accuracy == "medium":
            self.mc_num_samples, self.max_batch_size = 128, 128
        self.device = device

        if surrogate_model == "GP":
            self.surrogate_model_class = GPSurrogateModel
        elif surrogate_model == "RF":
            self.surrogate_model_class = RFSurrogateModel
        elif surrogate_model == "BNN":
            self.surrogate_model_class = BNNEnsembleSurrogateModel
        elif surrogate_model == "BayesianLinear":
            self.surrogate_model_class = BayesianLinearSurrogateModel
        else:
            raise ValueError(f"Unknown surrogate model: {surrogate_model}")

        if acq_func == "EHVI":
            self.acquisition_function_class = EHVIAcquisitionFunction
        elif acq_func == "UCB":
            self.acquisition_function_class = UCBAcquisitionFunction
        elif acq_func == "ParEGO":
            self.acquisition_function_class = ParEGOAcquisitionFunction
        elif acq_func == "NEI":
            self.acquisition_function_class = NEIAcquisitionFunction
        else:
            raise ValueError(f"Unknown acquisition function: {acq_func}")

        self.target_evaluator = ParetoFrontCalculator()

    def optimize(
        self,
        training_X: np.ndarray,
        training_y: np.ndarray,
        candidate_X: np.ndarray,
        opt_metric_settings: List[dict],
        batch_size: int,
        training_y_dict: dict,
        temperature: float = 0.0,
        constraints: dict = None,
        total_name_arr: np.ndarray = None,
        condition_types: List[str] = None,
        total_desc_arr: np.ndarray = None,
    ) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:

        training_X_t = torch.tensor(training_X).double().to(device=self.device)
        training_y_t = torch.tensor(training_y).double()
        training_y_t = self._weight_y(training_y_t, opt_metric_settings).to(device=self.device)
        candidate_X_t = torch.tensor(candidate_X).double().to(device=self.device)

        # with Progress(
        #     TextColumn("[bold cyan]{task.description}"),
        #     BarColumn(bar_width=None),
        #     MofNCompleteColumn(),
        #     TimeRemainingColumn(),
        #     console=self.console,
        # ) as progress:
        num_models = training_y_t.shape[1]
        # task_train = progress.add_task(description="Training surrogate models", total=num_models, start=True)

        models = []
        for i in range(num_models):
            key = list(training_y_dict.keys())[i]
            #   progress.log(f"Fitting model for [bold]{key}[/bold]...", style="yellow")

            train_y_i = training_y_t[:, i].reshape(-1, 1)
            model_i = self.surrogate_model_class(device=self.device, num_dims=training_X_t.shape[1])

            if isinstance(model_i, GPSurrogateModel):
                model_i.fit(training_X_t, train_y_i)
                models.append(model_i.model)
            else:
                wrapper = SklearnModelWrapper(model_i)
                wrapper.fit_surrogate(training_X_t, train_y_i)
                models.append(wrapper)

            # progress.update(task_train, advance=1)

        self.global_model = ModelListGP(*models)

        # task_pareto = progress.add_task(description="Calculating Pareto frontiers", total=len(training_y) - 1)
        training_y_np = training_y_t.cpu().numpy()
        # self.pareto_y = self.target_evaluator.calculate_target_function(training_y_np, progress, task_pareto).to(device=self.device)
        self.pareto_y = self.target_evaluator.calculate_target_function(training_y_np).to(device=self.device)

        self.ref_point = torch.tensor(
            [-1 if omi["opt_direct"] == "min" else 0 for omi in opt_metric_settings], dtype=float, device=self.device
        )

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([self.mc_num_samples]), seed=self.random_seed)
        if self.acquisition_function_class == EHVIAcquisitionFunction:
            partitioning = NondominatedPartitioning(ref_point=self.ref_point, Y=self.pareto_y)
            acq_func = self.acquisition_function_class(
                model=self.global_model,
                sampler=sampler,
                ref_point=self.ref_point,
                partitioning=partitioning,
            )
        elif self.acquisition_function_class == UCBAcquisitionFunction:
            weights = torch.tensor([d["metric_weight"] for d in opt_metric_settings], dtype=torch.double, device=self.device)
            acq_func = self.acquisition_function_class(
                model=self.global_model,
                sampler=sampler,
                beta=2.0,
                weights=weights,
            )
        elif self.acquisition_function_class == ParEGOAcquisitionFunction:

            acq_func = self.acquisition_function_class(
                model=self.global_model, sampler=sampler, X_baseline=training_X_t, num_objectives=len(opt_metric_settings)
            )
        elif self.acquisition_function_class == NEIAcquisitionFunction:
            weights = torch.tensor([d["metric_weight"] for d in opt_metric_settings], dtype=torch.double, device=self.device)
            acq_func = self.acquisition_function_class(model=self.global_model, sampler=sampler, X_baseline=training_X_t, weights=weights)
        else:
            raise ValueError(f"Unknown acquisition function class: {self.acquisition_function_class}")

        # Generate constraint mask if constraints are provided
        constraint_mask_t = None

        if constraints is not None and total_name_arr is not None and condition_types is not None:
            constraint_mask = generate_constraint_mask(
                total_name_arr=total_name_arr,
                condition_types=condition_types,
                constraints=constraints,
            )
            constraint_mask_t = torch.tensor(constraint_mask, dtype=torch.bool, device=self.device)
            # progress.log(f"Constraints applied: {constraint_mask.sum()}/{len(constraint_mask)} candidates available", style="cyan")

        # Generate unused reagent boost if total_name_arr is provided
        unused_reagent_boost = None
        if total_name_arr is not None and condition_types is not None and total_desc_arr is not None:
            unused_reagent_boost = self._compute_unused_reagent_boost(
                training_X=training_X_t,
                candidate_X=candidate_X_t,
                total_name_arr=total_name_arr,
                total_desc_arr=total_desc_arr,
                condition_types=condition_types,
                device=self.device,
            )

        # task_acq_opt = progress.add_task(description="Optimizing acquisition function", total=batch_size)
        self.acq_result, self.acq_value = acq_func.optimize_acqf_discrete(
            q=batch_size,
            choices=candidate_X_t,
            max_batch_size=self.max_batch_size,
            unique=True,
            exclude_points=training_X_t,
            min_distance=1e-6,
            # progress=progress,
            # task=task_acq_opt,
            temperature=temperature,
            constraint_mask=constraint_mask_t,
            unused_reagent_boost=unused_reagent_boost,
        )

        if self.device.type == "cuda":
            best_samples = [res.cpu().numpy() for res in self.acq_result]
        else:
            best_samples = [res.numpy() for res in self.acq_result]
        
        recommend_type = self._get_exploit_or_explore()
        pred_mean, pred_std = self._get_predictions()

        for i, d in enumerate(opt_metric_settings):
            if d["opt_direct"] == "min":
                pred_mean[:, i] = -pred_mean[:, i]

        pred_mean = self._unweight_y(torch.tensor(pred_mean), opt_metric_settings).numpy()
        pred_std = self._unweight_y(torch.tensor(pred_std), opt_metric_settings).numpy()
        return best_samples, recommend_type, pred_mean, pred_std

    def _get_exploit_or_explore(self) -> List[str]:
        with torch.no_grad():
            posterior = self.global_model.posterior(self.acq_result)
            pred_mean = posterior.mean

        # Handle different shapes from posterior.mean
        # The posterior.mean from ModelListGP with multiple outputs has shape (batch_size, q)
        # where each model contributes one dimension
        # We need to reshape to (batch_size, n_outputs) where batch_size = n_points
        if pred_mean.dim() == 2:
            # Already in correct shape (n_points, n_outputs)
            pass
        elif pred_mean.dim() == 3:
            # Shape: (1, n_points, n_outputs) -> (n_points, n_outputs)
            pred_mean = pred_mean.squeeze(0)

        # Ensure pred_mean is 2D
        if pred_mean.dim() != 2:
            raise ValueError(f"Unexpected pred_mean shape: {pred_mean.shape}")

        hvi_values = torch.tensor([compute_hvi(pred_mean[i], self.pareto_y, self.ref_point) for i in range(pred_mean.shape[0])])
        ehvi_values = self.acq_value.to(device="cpu")
        exploit_scores = hvi_values / (ehvi_values + 1e-6)
        explore_scores = 1 - exploit_scores

        for i in range(self.acq_result.shape[0]):
            self.console.log(
                f"Point {i}: "
                f"EHVI = {ehvi_values[i]:.3f}, "
                f"HVI = {hvi_values[i]:.3f}, "
                f"Exploit Score = {exploit_scores[i]:.3f}, "
                f"Explore Score = {explore_scores[i]:.3f}"
            )
        return ["exploit" if exploit_scores[i] > explore_scores[i] else "explore" for i in range(self.acq_result.shape[0])]

    def _get_predictions(self) -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            posterior = self.global_model.posterior(self.acq_result)
            pred_mean = posterior.mean
            pred_var = posterior.variance

        # Handle different shapes from posterior
        if pred_mean.dim() == 2:
            # Already in correct shape (n_points, n_outputs)
            pass
        elif pred_mean.dim() == 3:
            # Shape: (1, n_points, n_outputs) -> (n_points, n_outputs)
            pred_mean = pred_mean.squeeze(0)
            pred_var = pred_var.squeeze(0)

        # Ensure correct shapes
        if pred_mean.dim() != 2:
            raise ValueError(f"Unexpected pred_mean shape: {pred_mean.shape}")

        pred_mean = pred_mean.cpu().numpy()
        pred_var = pred_var.cpu().numpy()
        pred_std = np.sqrt(pred_var)
        return pred_mean, pred_std

    def _weight_y(self, training_y: torch.Tensor, opt_metric_settings: List[dict]) -> torch.Tensor:
        weights = torch.tensor([d["metric_weight"] for d in opt_metric_settings])
        training_y = training_y * weights
        return training_y

    def _unweight_y(self, training_y: torch.Tensor, opt_metric_settings: List[dict]) -> torch.Tensor:
        weights = torch.tensor([d["metric_weight"] for d in opt_metric_settings])
        training_y = training_y / weights
        return training_y

    def _compute_unused_reagent_boost(
        self,
        training_X: torch.Tensor,
        candidate_X: torch.Tensor,
        total_name_arr: np.ndarray,
        total_desc_arr: np.ndarray,
        condition_types: List[str],
        device: torch.device,
        boost_value: float = 0.5,
    ) -> torch.Tensor:
        """
        Compute boost values for candidates containing unused reagents.

        This method identifies which reagents have already been used in training data
        and gives a boost to candidates that contain unused reagents, encouraging
        exploration of the reagent space.

        Args:
            training_X: Training data descriptors (n_training, n_features)
            candidate_X: Candidate descriptors (n_candidates, n_features)
            total_name_arr: Array of all condition name combinations (n_total, n_condition_types)
            total_desc_arr: Array of all condition descriptors (n_total, n_features)
            condition_types: List of condition type names
            device: PyTorch device
            boost_value: Value to add to acquisition function for candidates with unused reagents (default: 0.5)

        Returns:
            Tensor of boost values for each candidate (n_candidates,)
        """
        # Convert tensors to numpy for easier comparison
        training_X_np = training_X.cpu().numpy()
        candidate_X_np = candidate_X.cpu().numpy()
        
        # Find training indices by matching descriptors in total_desc_arr
        training_indices = set()
        for train_desc in training_X_np:
            matches = np.all(np.abs(total_desc_arr - train_desc) < 1e-6, axis=1)
            matching_indices = np.where(matches)[0]
            if len(matching_indices) > 0:
                training_indices.add(matching_indices[0])
        
        # Get used reagents for each condition type
        used_reagents = {ct: set() for ct in condition_types}
        for idx in training_indices:
            for j, ct in enumerate(condition_types):
                used_reagents[ct].add(str(total_name_arr[idx, j]))
        
        # Log which reagents are used/unused
        self.console.print("[bold cyan]Reagent usage analysis:[/bold cyan]")
        for j, ct in enumerate(condition_types):
            all_reagents = set(str(x) for x in total_name_arr[:, j])
            unused = all_reagents - used_reagents[ct]
            self.console.print(f"  {ct}: {len(used_reagents[ct])} used, {len(unused)} unused", style="cyan")
        
        # Find candidate indices and compute boosts
        boosts = []
        for cand_desc in candidate_X_np:
            # Find matching index in total_desc_arr
            matches = np.all(np.abs(total_desc_arr - cand_desc) < 1e-6, axis=1)
            matching_indices = np.where(matches)[0]
            
            if len(matching_indices) == 0:
                # No match found, no boost
                boosts.append(0.0)
                continue
            
            cand_idx = matching_indices[0]
            
            # Check if this candidate contains any unused reagent
            has_unused_reagent = False
            for j, ct in enumerate(condition_types):
                reagent_name = str(total_name_arr[cand_idx, j])
                if reagent_name not in used_reagents[ct]:
                    has_unused_reagent = True
                    break
            
            boosts.append(boost_value if has_unused_reagent else 0.0)
        
        # Count how many candidates get a boost
        num_boosted = sum(1 for b in boosts if b > 0)
        self.console.print(f"[bold green]{num_boosted}/{len(boosts)} candidates receive unused reagent boost (+{boost_value})[/bold green]")
        
        return torch.tensor(boosts, dtype=torch.double, device=device)
