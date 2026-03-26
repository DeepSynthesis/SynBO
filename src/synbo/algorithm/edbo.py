"""
EDBO+ (Experimental Design via Bayesian Optimization) Implementation

This module contains a complete implementation of EDBO+ algorithm based on:
- Botorch for Bayesian Optimization
- GPyTorch for Gaussian Process models
- Multi-objective optimization using EHVI/NoisyEHVI acquisition functions
"""

import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import gpytorch
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.priors import GammaPrior
from gpytorch.constraints import GreaterThan
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement, qNoisyExpectedHypervolumeImprovement
from botorch.models import SingleTaskGP, ModelListGP
from botorch.optim import optimize_acqf_discrete
from botorch.sampling import SobolQMCNormalSampler, IIDNormalSampler
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from ordered_set import OrderedSet

from scipy.stats import norm
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler

# Default torch settings
tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cpu"),
}


class EDBOStandardScaler:
    """
    Custom standard scaler for EDBO.
    """
    def __init__(self):
        pass

    def fit(self, x):
        self.mu = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)

    def transform(self, x):
        for obj in range(0, len(self.std)):
            if self.std[obj] == 0.0:
                self.std[obj] = 1e-6
        return (x - [self.mu]) / [self.std]

    def fit_transform(self, x):
        self.mu = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)

        for obj in range(0, len(self.std)):
            if self.std[obj] == 0.0:
                self.std[obj] = 1e-6
        return (x - [self.mu]) / [self.std]

    def inverse_transform(self, x):
        return x * [self.std] + [self.mu]

    def inverse_transform_var(self, x):
        std = np.asarray(self.std)
        return x * (std ** 2)


def build_and_optimize_model(train_x, train_y):
    """Builds GP model and optimizes it using GPyTorch."""
    
    gp_options = {
        "ls_prior1": 2.0,
        "ls_prior2": 0.2,
        "ls_prior3": 5.0,
        "out_prior1": 5.0,
        "out_prior2": 0.5,
        "out_prior3": 8.0,
        "noise_prior1": 1.5,
        "noise_prior2": 0.1,
        "noise_prior3": 5.0,
        "noise_constraint": 1e-5,
    }

    n_features = np.shape(train_x)[1]

    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()

            kernels = MaternKernel(
                ard_num_dims=n_features, 
                lengthscale_prior=GammaPrior(gp_options["ls_prior1"], gp_options["ls_prior2"])
            )

            self.covar_module = ScaleKernel(
                kernels, 
                outputscale_prior=GammaPrior(gp_options["out_prior1"], gp_options["out_prior2"])
            )
            try:
                ls_init = gp_options["ls_prior3"]
                self.covar_module.base_kernel.lengthscale = ls_init
            except:
                uniform = gp_options["ls_prior3"]
                ls_init = torch.ones(n_features).to(**tkwargs) * uniform
                self.covar_module.base_kernel.lengthscale = ls_init

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # Initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        GammaPrior(gp_options["noise_prior1"], gp_options["noise_prior2"])
    )

    likelihood.noise = gp_options["noise_prior3"]
    model = ExactGPModel(train_x, train_y, likelihood).to(**tkwargs)

    model.likelihood.noise_covar.register_constraint(
        "raw_noise", GreaterThan(gp_options["noise_constraint"])
    )

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}],
        lr=0.1,
    )

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    training_iter = 1000

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y.squeeze(-1).to(**tkwargs))
        loss.backward()
        optimizer.step()

    model.eval()
    likelihood.eval()
    return model, likelihood


class EDBOplus:
    """
    EDBO+ optimizer class for multi-objective Bayesian optimization.
    
    This class implements the full EDBO+ algorithm including:
    - Initialization sampling
    - Gaussian Process modeling
    - Acquisition function optimization (EHVI/NoisyEHVI)
    - Batch recommendation
    """

    def __init__(self):
        self.predicted_mean = []
        self.predicted_variance = []
        self.ei = None
        self.acquisition_sampler = "SobolQMCNormalSampler"

    @staticmethod
    def _init_sampling(df, batch, sampling_method, seed, init_indices=None):
        """
        Initialize sampling for the first batch of experiments.
        
        Args:
            df: DataFrame with reaction scope
            batch: Number of samples to select
            sampling_method: Method for sampling ('random', 'with_index', etc.)
            seed: Random seed
            init_indices: Optional list of indices to use for initialization
            
        Returns:
            DataFrame with 'priority' column indicating selected samples
        """
        np.random.seed(seed)
        random.seed(seed)
        numeric_cols = df._get_numeric_data().columns
        ohe_columns = list(OrderedSet(df.columns) - OrderedSet(numeric_cols))
        if len(ohe_columns) > 0:
            print(f"The following columns are categorical and will be encoded using One-Hot-Encoding: {ohe_columns}")
        
        # Encode OHE
        df_sampling = pd.get_dummies(df, prefix=ohe_columns, columns=ohe_columns, drop_first=True, dtype=np.float64)

        class HiddenPrints:
            def __enter__(self):
                self._original_stdout = sys.stdout
                sys.stdout = open(os.devnull, "w")

            def __exit__(self, exc_type, exc_val, exc_tb):
                sys.stdout.close()
                sys.stdout = self._original_stdout

        # Order df according to initial sampling method
        with HiddenPrints():
            idaes = None
            samples = None

            if sampling_method == "with_index" and init_indices is not None:
                # Use provided indices for initialization
                samples = df_sampling.iloc[init_indices]
            elif sampling_method == "random":
                samples = df_sampling.sample(n=batch, random_state=seed)
            else:
                # Default to random if unknown method
                samples = df_sampling.sample(n=batch, random_state=seed)

            if idaes is not None:
                samples = idaes.sample_points()
                if not isinstance(samples, pd.DataFrame):
                    samples = pd.DataFrame(samples)

            # Add random samples if not enough
            if sampling_method != "with_index" and len(samples) < batch:
                additional_samples = df.sample(n=batch - len(samples), random_state=seed, replace=True)
                additional_samples = additional_samples.reset_index(drop=True)

            # Add additional samples until batch size is reached
            extra_seed = 1
            if sampling_method != "with_index":
                while len(samples) < batch:
                    samples = pd.concat([samples, additional_samples]).drop_duplicates(ignore_index=True)
                    additional_samples = df.sample(n=batch - len(samples), random_state=seed + extra_seed, replace=True)
                    extra_seed += 1

        # Get index of the best samples
        df_sampling_matrix = df_sampling.to_numpy()
        priority_list = np.zeros_like(df_sampling.index)

        for sample in samples.to_numpy():
            d_i = cdist([sample], df_sampling_matrix, metric="cityblock")
            a = np.argmin(d_i)
            priority_list[a] = 1.0
        
        df["priority"] = priority_list

        if sampling_method == "with_index" and init_indices is not None:
            print(f"Initialized with {len(init_indices)} provided indices")
        else:
            print(f"Generated {len(samples)} initial samples using {sampling_method} sampling (seed = {seed}).")

        return df

    def run(
        self,
        objectives,
        objective_mode,
        objective_thresholds=None,
        directory=".",
        filename="reaction.csv",
        columns_features="all",
        batch=5,
        init_sampling_method="random",
        seed=0,
        scaler_features=MinMaxScaler(),
        scaler_objectives=EDBOStandardScaler(),
        acquisition_function="NoisyEHVI",
        acquisition_function_sampler="SobolQMCNormalSampler",
        init_indices=None,
    ):
        """
        Run EDBO+ optimization.
        
        Args:
            objectives: List of objective column names
            objective_mode: List of 'max' or 'min' for each objective
            objective_thresholds: List of threshold values for each objective
            directory: Directory containing the scope file
            filename: Name of the scope CSV file
            columns_features: List of feature columns to use ('all' for auto)
            batch: Batch size (number of experiments to recommend)
            init_sampling_method: Method for initial sampling
            seed: Random seed
            scaler_features: Scaler for features
            scaler_objectives: Scaler for objectives
            acquisition_function: Acquisition function ('EHVI', 'NoisyEHVI', 'EI')
            acquisition_function_sampler: Sampler for acquisition function
            init_indices: Optional indices for initialization
            
        Returns:
            DataFrame with recommendations and priorities
        """
        wdir = Path(directory)
        csv_filename = wdir.joinpath(filename)
        torch.manual_seed(seed=seed)
        np.random.seed(seed)
        self.acquisition_sampler = acquisition_function_sampler

        # 1. Safe checks
        self.objective_names = objectives
        
        # Check whether the columns_features contains the objectives
        if columns_features != "all":
            for objective in objectives:
                if objective in columns_features:
                    columns_features.remove(objective)
                if "priority" in columns_features:
                    columns_features.remove("priority")

        # Check objectives is a list
        if type(objectives) != list:
            objectives = [objectives]
        if type(objective_mode) != list:
            objective_mode = [objective_mode]

        # Check that the user's scope exists
        msg = "Scope was not found. Please create a scope (csv file)."
        assert os.path.exists(csv_filename), msg

        # 2. Load reaction scope
        df = pd.read_csv(f"{csv_filename}")
        df = df.dropna(axis="columns", how="all")
        df.drop(["base", "ligand", "solvent", "index", "temperature", "concentration"], axis=1, inplace=True)
        original_df = df.copy(deep=True)

        # 2.1. Initialize sampling (only in the first iteration)
        obj_in_df = list(filter(lambda x: x in df.columns.values, objectives))

        # Check whether new objective has been added – if not add PENDING
        for obj_i in self.objective_names:
            if obj_i not in original_df.columns.values:
                original_df[obj_i] = ["PENDING"] * len(original_df.values)

        if columns_features != "all":
            if "priority" in df.columns.values:
                for obj_i in objectives:
                    if obj_i not in df.columns.values:
                        df[obj_i] = ["PENDING"] * len(df.values)
                df = df[columns_features + objectives + ["priority"]]
            else:
                if len(obj_in_df) == 0:
                    df = df[columns_features]
                else:
                    df = df[columns_features + objectives]

        # No objectives columns in the scope? Then random initialization
        if len(obj_in_df) == 0:
            print("There are no experimental observations yet. Random samples will be drawn.")
            df = self._init_sampling(
                df=df, batch=batch, seed=seed, 
                sampling_method=init_sampling_method, 
                init_indices=init_indices
            )
            original_df["priority"] = df["priority"]
            # Append objectives
            for objective in objectives:
                if objective not in original_df.columns.values:
                    original_df[objective] = ["PENDING"] * len(original_df)

            # Sort values and save dataframe
            original_df = original_df.sort_values("priority", ascending=False)
            original_df = original_df.loc[:, ~original_df.columns.str.contains("^Unnamed")]
            original_df.to_csv(csv_filename, index=False)
            return original_df

        if columns_features == "all":
            columns_features = list(set(df.columns.tolist()) - set(objectives) - set(["priority"]))
        
        print(f"This run will optimize for the following objectives: {objectives}")
        print(f"The following features will be used: {columns_features[:5]}...")

        # 3. Separate train and test data

        # 3.1. Auto-detect dummy features (one-hot-encoding)
        numeric_cols = df._get_numeric_data().columns
        for nc in numeric_cols:
            df[nc] = pd.to_numeric(df[nc], downcast="float")
        ohe_columns = list(OrderedSet(df.columns) - OrderedSet(numeric_cols))
        ohe_columns = list(OrderedSet(ohe_columns) - OrderedSet(objectives))

        ohe_features = False
        if len(ohe_columns) > 0:
            print(f"The following columns are categorical and will be encoded using One-Hot-Encoding: {ohe_columns}")
            ohe_features = True

        data = pd.get_dummies(df, prefix=ohe_columns, columns=ohe_columns, drop_first=True, dtype=np.float64)

        # 3.2. Any sample with a value 'PENDING' in any objective is a test
        idx_test = (data[data.apply(lambda r: r.astype(str).str.contains("PENDING", case=False).any(), axis=1)]).index.values
        idx_train = (data[~data.apply(lambda r: r.astype(str).str.contains("PENDING", case=False).any(), axis=1)]).index.values

        # Data only contains featurized information (train and test)
        df_train_y = data.loc[idx_train][objectives]
        if "priority" in data.columns.tolist():
            data = data.drop(columns=objectives + ["priority"])
        else:
            data = data.drop(columns=objectives)
        
        df_train_x = data.loc[idx_train]
        df_test_x = data.loc[idx_test]

        if len(df_train_x.values) == 0:
            msg = "The scope was already generated, please insert at least one experimental observation value and then press run."
            print(msg)
            return original_df

        # Run the BO process
        priority_list = self._model_run(
            data=data,
            df_train_x=df_train_x,
            df_test_x=df_test_x,
            df_train_y=df_train_y,
            batch=batch,
            objective_mode=objective_mode,
            objective_thresholds=objective_thresholds,
            seed=seed,
            scaler_x=scaler_features,
            scaler_y=scaler_objectives,
            acquisition_function=acquisition_function,
        )

        # Low priority to the samples that have been already collected
        for i in range(0, len(idx_train)):
            priority_list[idx_train[i]] = -1

        original_df["priority"] = priority_list

        cols_sort = ["priority"] + original_df.columns.values.tolist()
        
        # Attach objectives predictions and expected improvement
        cols_for_preds = []
        for idx_obj in range(0, len(objectives)):
            name = objectives[idx_obj]
            mean = self.predicted_mean[:, idx_obj]
            var = self.predicted_variance[:, idx_obj]
            std_dev = np.sqrt(var)
            ei = self.ei[:, idx_obj]
            original_df[f"{name}_predicted_mean"] = mean
            original_df[f"{name}_predicted_std_dev"] = std_dev
            original_df[f"{name}_expected_improvement"] = ei
            cols_for_preds.append([f"{name}_predicted_mean", f"{name}_predicted_std_dev", f"{name}_expected_improvement"])
        cols_for_preds = np.ravel(cols_for_preds)

        original_df = original_df.sort_values(cols_sort, ascending=False)
        # Save extra df containing predictions, uncertainties and EI
        original_df.to_csv(f"{directory}/pred_{filename}", index=False)
        # Drop predictions, uncertainties and EI
        original_df = original_df.drop(columns=cols_for_preds, axis="columns")
        original_df = original_df.sort_values(cols_sort, ascending=False)
        original_df.to_csv(csv_filename, index=False)

        print("Run finished!")
        return original_df

    def _model_run(
        self,
        data,
        df_train_x,
        df_test_x,
        df_train_y,
        batch,
        objective_mode,
        objective_thresholds,
        seed,
        scaler_x,
        scaler_y,
        acquisition_function,
    ):
        """
        Runs the surrogate machine learning model.
        Returns a priority list for a given scope (top priority to low priority).
        """
        # Check number of objectives
        n_objectives = len(df_train_y.columns.values)

        scaler_x.fit(df_train_x.to_numpy())
        init_train = scaler_x.transform(df_train_x.to_numpy())
        test_xnp = scaler_x.transform(df_test_x.to_numpy())
        test_x = torch.tensor(test_xnp.tolist()).double().to(**tkwargs)
        y = df_train_y.astype(float).to_numpy()  # not scaled

        individual_models = []
        for i in range(0, n_objectives):
            if objective_mode[i].lower() == "min":
                y[:, i] = -y[:, i]
        y = scaler_y.fit_transform(y)

        print("Generating surrogate model...")
        for i in range(0, n_objectives):
            train_x = torch.tensor(init_train).to(**tkwargs).double()
            train_y = np.array(y)[:, i]
            train_y = np.atleast_2d(train_y).reshape(len(train_y), -1)
            train_y_i = torch.tensor(train_y.tolist()).to(**tkwargs).double()

            gp, likelihood = build_and_optimize_model(train_x=train_x, train_y=train_y_i)

            model_i = SingleTaskGP(
                train_X=train_x, 
                train_Y=train_y_i, 
                covar_module=gp.covar_module, 
                likelihood=likelihood
            )
            individual_models.append(model_i)

        print("Model generated!")

        # Reference point is the minimum seen so far
        ref_mins = np.min(y, axis=0)
        if objective_thresholds is None:
            ref_point = torch.tensor(ref_mins).double().to(**tkwargs)
        else:
            ref_point = np.zeros(n_objectives)
            for i in range(0, n_objectives):
                if objective_thresholds[i] is None:
                    ref_point[i] = ref_mins[i]
                else:
                    ref_point[i] = objective_thresholds[i]
                    if objective_mode[i].lower() == "min":
                        ref_point[i] = -ref_point[i]
            # Scale
            ref_point = scaler_y.transform(np.array([ref_point]))
            # Loop again
            for i in range(0, n_objectives):
                if objective_thresholds[i] is None:
                    ref_point[0][i] = ref_mins[i]
            ref_point = torch.tensor(ref_point[0]).double().to(**tkwargs)

        # Determine number of samples based on data size
        if len(data.values) > 100000:
            sobol_num_samples = 64
        elif len(data.values) > 50000:
            sobol_num_samples = 128
        elif len(data.values) > 10000:
            sobol_num_samples = 256
        else:
            sobol_num_samples = 512

        y_torch = torch.tensor(y).to(**tkwargs).double()

        if self.acquisition_sampler == "IIDNormalSampler":
            sampler = IIDNormalSampler(num_samples=sobol_num_samples, collapse_batch_dims=True, seed=seed)
        if self.acquisition_sampler == "SobolQMCNormalSampler":
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([sobol_num_samples]), seed=seed)

        print("Optimizing acquisition function...")

        surrogate_model = None

        if acquisition_function.lower() == "ehvi":
            partitioning = NondominatedPartitioning(ref_point=ref_point, Y=y_torch)

            surrogate_model = ModelListGP(*individual_models)
            individual_models = []  # empty to reduce memory

            EHVI = qExpectedHypervolumeImprovement(
                model=surrogate_model,
                sampler=sampler,
                ref_point=ref_point,
                partitioning=partitioning
            )

            print(ref_point)

            acq_result = optimize_acqf_discrete(acq_function=EHVI, choices=test_x, q=batch, unique=True)

        if acquisition_function.lower() == "noisyehvi":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                acq_fct = None
                if n_objectives > 1:
                    surrogate_model = ModelListGP(*individual_models)
                    train_x = torch.tensor(init_train).to(**tkwargs).double()
                    acq_fct = qNoisyExpectedHypervolumeImprovement(
                        model=surrogate_model,
                        sampler=sampler,
                        ref_point=ref_point,
                        alpha=0.0,
                        incremental_nehvi=True,
                        X_baseline=train_x,
                        prune_baseline=True,
                    )
                else:
                    surrogate_model = individual_models[0]
                    best_value = y_torch.max()
                    acq_fct = qExpectedImprovement(model=surrogate_model, best_f=best_value, sampler=sampler)

                acq_result = optimize_acqf_discrete(acq_function=acq_fct, choices=test_x, q=batch, unique=True)

        best_samples = scaler_x.inverse_transform(acq_result[0].detach().cpu().numpy())

        print("Acquisition function optimized.")

        # Save rescaled predictions (only for first fantasy)

        # Get predictions in chunks
        chunk_size = 1000
        n_chunks = len(data.values) // chunk_size

        if n_chunks == 0:
            n_chunks = 1

        self.predicted_mean = np.zeros(shape=(len(data.values), n_objectives))
        self.predicted_variance = np.zeros(shape=(len(data.values), n_objectives))
        self.ei = np.zeros(shape=(len(data.values), n_objectives))

        observed_raw_values = df_train_y.astype(float).to_numpy()

        for i in range(0, len(data.values), n_chunks):
            vals = data.values[i : i + n_chunks]
            data_tensor = torch.tensor(scaler_x.transform(vals)).double().to(**tkwargs)
            preds = surrogate_model.posterior(X=data_tensor)
            self.predicted_mean[i : i + n_chunks] = scaler_y.inverse_transform(preds.mean.detach().cpu().numpy())
            self.predicted_variance[i : i + n_chunks] = scaler_y.inverse_transform_var(preds.variance.detach().cpu().numpy())

            for j in range(0, len(objective_mode)):
                maximizing = False
                if objective_mode[j] == "max":
                    maximizing = True
                self.ei[i : i + n_chunks, j] = self.expected_improvement(
                    train_y=observed_raw_values[:, j],
                    mean=self.predicted_mean[i : i + n_chunks, j],
                    variance=self.predicted_variance[i : i + n_chunks, j],
                    maximizing=maximizing,
                )

        print("Predictions and expected improvement obtained.")

        # Flip predictions if needed
        for i in range(0, len(objective_mode)):
            if objective_mode[i] == "min":
                self.predicted_mean[:, i] = -self.predicted_mean[:, i]

        # Rescale samples
        all_samples = data.values

        priority_list = [0] * len(data.values)

        # Find best samples in data
        for sample in best_samples:
            d_i = cdist([sample], all_samples, metric="cityblock")
            a = np.argmin(d_i)
            priority_list[a] = 1.0

        return priority_list

    def expected_improvement(self, train_y, mean, variance, maximizing=False):
        """
        Expected improvement acquisition function.
        
        Args:
            train_y: Numpy array with observed train targets
            mean: Predicted mean of the Gaussian Process
            variance: Predicted variance of the Gaussian Process
            maximizing: Whether to maximize or minimize
            
        Returns:
            Expected improvement values
        """
        sigma = np.sqrt(variance)

        if maximizing:
            loss_optimum = np.max(train_y)
        else:
            loss_optimum = np.min(train_y)

        scaling_factor = (-1) ** (not maximizing)

        # In case sigma equals zero
        with np.errstate(divide="ignore"):
            Z = scaling_factor * (mean - loss_optimum) / sigma
            expected_improvement = scaling_factor * (mean - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
            expected_improvement[sigma == 0.0] = 0.0

        return expected_improvement