import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesRegressor
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
import warnings

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")


def load_data():
    """Load main dataset and descriptor files"""
    df = pd.read_csv("Co-enamine.csv")

    # Load descriptor files (already combined RDKit + DFT)
    desc_files = {
        "amine": "descriptors/amine_desc.csv",
        "cobalt": "descriptors/cobalt_desc.csv",
        "oxidant": "descriptors/oxidant_desc.csv",
        "alkali": "descriptors/alkali_desc.csv",
        "solvent": "descriptors/solvent_desc.csv",
    }

    descriptors = {}
    for col, filepath in desc_files.items():
        try:
            desc_df = pd.read_csv(filepath)
            desc_df = desc_df.set_index("SMILES")
            descriptors[col] = desc_df
            print(f"  Loaded descriptors for {col}: {desc_df.shape[1]} features")
        except Exception as e:
            print(f"  Warning: Could not load descriptors for {col}: {e}")
            descriptors[col] = None

    return df, descriptors


def generate_features(df, descriptors):
    """Generate features from loaded descriptor files"""
    feature_list = []
    smiles_cols = ["amine", "cobalt", "oxidant", "alkali", "solvent"]

    print("Generating features from descriptor files...")

    for smiles_col in smiles_cols:
        print(f"  Processing {smiles_col}...")

        if descriptors[smiles_col] is not None:
            desc_df = descriptors[smiles_col]
            features = []
            for smiles in df[smiles_col]:
                if smiles in desc_df.index:
                    features.append(desc_df.loc[smiles].values)
                else:
                    features.append(np.zeros(desc_df.shape[1]))

            features = np.array(features)
            # Prefix column names to avoid conflicts
            feat_cols = [f"{smiles_col}_{col}" for col in desc_df.columns]
            feature_df = pd.DataFrame(data=features, columns=feat_cols)
            print(f"    Loaded {features.shape[1]} features")
        else:
            print(f"    Warning: No descriptors available for {smiles_col}")
            feature_df = pd.DataFrame()

        feature_list.append(feature_df)

    X = pd.concat(feature_list, axis=1)
    return X


def cv_validation(model, X, y, cv, scoring="r2"):
    """Custom cross-validation function to avoid nested parallelism issues.

    Performs K-Fold cross-validation sequentially to prevent resource competition
    between cross_val_score's n_jobs and model's n_jobs.

    Parameters
    ----------
    model : estimator
        The model to validate
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    cv : cross-validation splitter
        e.g., KFold object
    scoring : str, default="r2"
        Scoring metric, currently only supports "r2"

    Returns
    -------
    scores : ndarray
        Array of scores for each fold
    """
    from sklearn.metrics import get_scorer

    scorer = get_scorer(scoring)
    scores = []

    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Clone model to ensure fresh instance for each fold
        from sklearn.base import clone

        fold_model = clone(model)

        # Fit and predict
        fold_model.fit(X_train, y_train)
        y_pred = fold_model.predict(X_val)

        # Calculate score
        score = scorer._score_func(y_val, y_pred)
        scores.append(score)

    return np.array(scores)


def objective_yield_optuna(trial, X, y, y_aux):
    """Objective function for yield optimization using Optuna"""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Define hyperparameters
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 3000, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "n_features": trial.suggest_int("n_features", 100, 2000, step=50),
    }
    # from IPython import embed; embed()

    # Feature selection
    n_features = params["n_features"]
    selector = SelectFromModel(ExtraTreesRegressor(n_estimators=50, random_state=42), max_features=n_features, threshold=-np.inf)

    # Add auxiliary feature (ee)
    X_with_aux = np.hstack([X, y_aux.reshape(-1, 1)])
    X_sel = selector.fit_transform(X_with_aux, y)

    # Create XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        min_child_weight=params["min_child_weight"],
        gamma=params["gamma"],
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    # 5-fold CV using custom cv_validation
    cv_scores = cv_validation(model, X_sel, y, cv=kf, scoring="r2")
    cv_r2_mean = cv_scores.mean()

    return cv_r2_mean


def objective_ee_optuna(trial, X, y, y_aux):
    """Objective function for ee optimization using Optuna"""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Define hyperparameters
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 3000, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "n_features": trial.suggest_int("n_features", 100, 2000, step=50),
    }

    try:
        # Feature selection
        n_features = params["n_features"]
        print("select feature")
        selector = SelectFromModel(ExtraTreesRegressor(n_estimators=200, random_state=42), max_features=n_features, threshold=-np.inf)

        # Add auxiliary feature (yield)
        X_with_aux = np.hstack([X, y_aux.reshape(-1, 1)])
        X_sel = selector.fit_transform(X_with_aux, y)

        # Create XGBoost model
        model = xgb.XGBRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            reg_alpha=params["reg_alpha"],
            reg_lambda=params["reg_lambda"],
            min_child_weight=params["min_child_weight"],
            gamma=params["gamma"],
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

        # 5-fold CV using custom cv_validation
        print("training model")
        cv_scores = cv_validation(model, X_sel, y, cv=kf, scoring="r2")
        cv_r2_mean = cv_scores.mean()

        return cv_r2_mean

    except Exception as e:
        return -1.0


def optimize_hyperparameters_optuna(objective_fn, X, y, y_aux, n_trials=50):
    """Run Optuna optimization"""
    print(f"\nStarting hyperparameter optimization with {n_trials} trials...")

    # Create a study with TPE sampler
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))

    # Optimize
    study.optimize(lambda trial: objective_fn(trial, X, y, y_aux), n_trials=n_trials, show_progress_bar=True)

    # Get best parameters
    best_params = study.best_params
    best_cv_r2 = study.best_value

    return best_params, best_cv_r2, study


def plot_cv_predictions(y_true, y_pred, target_name, r2, filename):
    """Plot predicted vs true values from CV predictions"""
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.6, s=60, edgecolor="none")

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")

    plt.xlabel("True Values", fontsize=14, fontweight="bold")
    plt.ylabel("Predicted Values", fontsize=14, fontweight="bold")
    plt.title(f"{target_name}: 5-Fold CV Predicted vs True\nR² = {r2:.4f}", fontsize=16, fontweight="bold")

    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"  Plot saved: {filename}")
    plt.close()


def train_and_evaluate_best_model(X, y, params, target_name):
    """Train model with best parameters and evaluate with 5-fold CV"""
    print(f"\n  Feature selection with n_features={int(params['n_features'])}...")
    selector = SelectFromModel(
        ExtraTreesRegressor(n_estimators=200, random_state=42), max_features=int(params["n_features"]), threshold=-np.inf
    )
    X_sel = selector.fit_transform(X, y)
    print(f"  Selected {X_sel.shape[1]} features")

    print(f"\n  Training XGBoost model for {target_name} with best params...")

    # Create model with best params
    model = xgb.XGBRegressor(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        min_child_weight=int(params["min_child_weight"]),
        gamma=params["gamma"],
        random_state=42,
        n_jobs=-1,
    )

    # 5-fold CV using custom cv_validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cv_validation(model, X_sel, y, cv=kf, scoring="r2")
    cv_r2_mean = cv_scores.mean()
    cv_r2_std = cv_scores.std()

    # Cross-validated predictions for plotting
    y_pred = cross_val_predict(model, X_sel, y, cv=kf, n_jobs=-1)
    y_pred = np.clip(y_pred, 0, 100)

    overall_r2 = r2_score(y, y_pred)

    print(f"    5-Fold CV R²: {cv_r2_mean:.4f} (+/- {cv_r2_std:.4f})")
    print(f"    Overall R² (all data): {overall_r2:.4f}")

    return {
        "cv_r2_mean": cv_r2_mean,
        "cv_r2_std": cv_r2_std,
        "overall_r2": overall_r2,
        "y_true": y,
        "y_pred": y_pred,
        "model": model,
        "X_sel": X_sel,
    }


def main():
    """Main function"""
    print("=" * 70)
    print("Hyperparameter Optimization with Optuna - XGBoost (5-Fold CV)")
    print("Target: CV R² >= 0.65 for both Yield and ee")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    df, descriptors = load_data()
    print(f"\nDataset: {df.shape[0]} samples")

    # Generate features
    print("\nGenerating Features...")
    X = generate_features(df, descriptors)
    print(f"\nTotal features: {X.shape[1]}")

    y_yield = df["yield"].values
    y_ee = df["ee"].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Scale auxiliary targets
    scaler_yield = StandardScaler()
    scaler_ee = StandardScaler()
    yield_scaled = scaler_yield.fit_transform(y_yield.reshape(-1, 1)).flatten()
    ee_scaled = scaler_ee.fit_transform(y_ee.reshape(-1, 1)).flatten()

    results = {}

    # ========== YIELD PREDICTION ==========
    print("\n" + "=" * 70)
    print("YIELD PREDICTION - Hyperparameter Optimization")
    print("=" * 70)

    # Use ee as auxiliary feature
    yield_best_params, yield_best_cv_r2, yield_study = optimize_hyperparameters_optuna(
        objective_yield_optuna, X_scaled, y_yield, ee_scaled, n_trials=50
    )

    print(f"\nBest CV R² for Yield: {yield_best_cv_r2:.4f}")
    print(f"Best parameters: {yield_best_params}")

    # Train and evaluate with best params
    X_yield = np.hstack([X_scaled, ee_scaled.reshape(-1, 1)])
    result_yield = train_and_evaluate_best_model(X_yield, y_yield, yield_best_params, "Yield")
    plot_cv_predictions(
        result_yield["y_true"], result_yield["y_pred"], "Yield (Optimized XGBoost)", result_yield["overall_r2"], "yield_optimized.png"
    )

    results["Yield"] = {
        "best_params": yield_best_params,
        "cv_r2_mean": result_yield["cv_r2_mean"],
        "cv_r2_std": result_yield["cv_r2_std"],
        "overall_r2": result_yield["overall_r2"],
    }

    # ========== EE PREDICTION ==========
    print("\n" + "=" * 70)
    print("ENANTIOMERIC EXCESS PREDICTION - Hyperparameter Optimization")
    print("=" * 70)

    # Use yield as auxiliary feature
    ee_best_params, ee_best_cv_r2, ee_study = optimize_hyperparameters_optuna(
        objective_ee_optuna, X_scaled, y_ee, yield_scaled, n_trials=50
    )

    print(f"\nBest CV R² for ee: {ee_best_cv_r2:.4f}")
    print(f"Best parameters: {ee_best_params}")

    # Train and evaluate with best params
    X_ee = np.hstack([X_scaled, yield_scaled.reshape(-1, 1)])
    result_ee = train_and_evaluate_best_model(X_ee, y_ee, ee_best_params, "ee")
    plot_cv_predictions(
        result_ee["y_true"], result_ee["y_pred"], "Enantiomeric Excess (Optimized XGBoost)", result_ee["overall_r2"], "ee_optimized.png"
    )

    results["ee"] = {
        "best_params": ee_best_params,
        "cv_r2_mean": result_ee["cv_r2_mean"],
        "cv_r2_std": result_ee["cv_r2_std"],
        "overall_r2": result_ee["overall_r2"],
    }

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY - Hyperparameter Optimization Results")
    print("=" * 70)

    # Yield results
    print("\nYield Prediction Results:")
    print(f"  5-Fold CV R²: {results['Yield']['cv_r2_mean']:.4f} (+/- {results['Yield']['cv_r2_std']:.4f})")
    print(f"  Overall R²:   {results['Yield']['overall_r2']:.4f}")
    print(f"  Status: {'✅ >= 0.65' if results['Yield']['cv_r2_mean'] >= 0.65 else '⚠️ < 0.65'}")

    # ee results
    print("\nEnantiomeric Excess Prediction Results:")
    print(f"  5-Fold CV R²: {results['ee']['cv_r2_mean']:.4f} (+/- {results['ee']['cv_r2_std']:.4f})")
    print(f"  Overall R²:   {results['ee']['overall_r2']:.4f}")
    print(f"  Status: {'✅ >= 0.65' if results['ee']['cv_r2_mean'] >= 0.65 else '⚠️ < 0.65'}")

    # Print optimized parameters
    print(f"\n{'='*70}")
    print("OPTIMIZED HYPERPARAMETERS:")
    print(f"\n{'='*70}")
    print("\nYield best params:")
    for key, value in results["Yield"]["best_params"].items():
        print(f"  {key}: {value}")

    print(f"\n{'='*70}")
    print("\nee best params:")
    for key, value in results["ee"]["best_params"].items():
        print(f"  {key}: {value}")

    # Overall status
    yield_ok = results["Yield"]["cv_r2_mean"] >= 0.65
    ee_ok = results["ee"]["cv_r2_mean"] >= 0.65

    print(f"\n{'='*70}")
    if yield_ok and ee_ok:
        print("🎉 SUCCESS! Both targets achieved 5-Fold CV R² >= 0.65!")
    else:
        print("⚠️  Some targets did not reach R² >= 0.65")
        if not yield_ok:
            print(f"   Yield gap: {0.65 - results['Yield']['cv_r2_mean']:.4f}")
        if not ee_ok:
            print(f"   ee gap: {0.65 - results['ee']['cv_r2_mean']:.4f}")
    print("=" * 70)

    # Save results to file
    with open("optimization_results.txt", "w") as f:
        f.write("=" * 70 + "\n")
        f.write("HYPERPARAMETER OPTIMIZATION RESULTS\n")
        f.write("=" * 70 + "\n\n")

        f.write("Yield Prediction:\n")
        f.write(f"  CV R²: {results['Yield']['cv_r2_mean']:.4f} (+/- {results['Yield']['cv_r2_std']:.4f})\n")
        f.write(f"  Best Parameters:\n")
        for key, value in results["Yield"]["best_params"].items():
            f.write(f"    {key}: {value}\n")

        f.write("\n" + "=" * 70 + "\n\n")

        f.write("Enantiomeric Excess Prediction:\n")
        f.write(f"  CV R²: {results['ee']['cv_r2_mean']:.4f} (+/- {results['ee']['cv_r2_std']:.4f})\n")
        f.write(f"  Best Parameters:\n")
        for key, value in results["ee"]["best_params"].items():
            f.write(f"    {key}: {value}\n")

    print("\nResults saved to: optimization_results.txt")


if __name__ == "__main__":
    main()
