import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")


def load_data():
    """Load main dataset and descriptor files"""
    df = pd.read_csv("Co-enamine.csv")
    
    desc_files = {
        "amine_smiles_ion": "descriptors/amine_ion_desc_dft.csv",
        "amine_smiles_anion": "descriptors/amine_anion_desc_dft.csv",
        "cobalt_smiles": "descriptors/cobalt_desc_dft.csv",
        "oxidant_smiles": "descriptors/oxidant_desc_dft.csv",
        "alkali_smiles": "descriptors/alkali_desc_dft.csv",
        "solvent_smiles": "descriptors/solvent_desc_dft.csv",
    }
    
    descriptors = {}
    for name, path in desc_files.items():
        descriptors[name] = pd.read_csv(path)
    
    return df, descriptors


def match_descriptors(df, descriptors, n_components=8):
    """Match descriptors with more PCA components"""
    feature_list = []
    
    for smiles_col, desc_df in descriptors.items():
        desc_dict = {}
        for idx, row in desc_df.iterrows():
            smiles = row["SMILES"]
            desc_cols = [col for col in desc_df.columns if col != "SMILES"]
            desc_dict[smiles] = row[desc_cols].values
        
        matched_desc = []
        for smiles in df[smiles_col]:
            if smiles in desc_dict:
                matched_desc.append(desc_dict[smiles])
            else:
                desc_cols = [col for col in desc_df.columns if col != "SMILES"]
                matched_desc.append([np.nan] * len(desc_cols))
        
        matched_desc = np.array(matched_desc)
        desc_df_matched = pd.DataFrame(matched_desc)
        desc_df_matched = desc_df_matched.fillna(desc_df_matched.mean())
        
        # Apply PCA with more components
        n_comp = min(n_components, matched_desc.shape[1])
        pca = PCA(n_components=n_comp, random_state=42)
        desc_pca = pca.fit_transform(desc_df_matched)
        
        pca_cols = [f"{smiles_col}_PC{i}" for i in range(n_comp)]
        desc_pca_df = pd.DataFrame(desc_pca, columns=pca_cols)
        feature_list.append(desc_pca_df)
        
        print(f"{smiles_col}: {matched_desc.shape[1]} -> {n_comp} dims "
              f"(var: {pca.explained_variance_ratio_.sum():.4f})")
    
    X = pd.concat(feature_list, axis=1)
    return X


def plot_predictions(y_true, y_pred, target_name, r2, mae, rmse, model_name=""):
    """Plot predicted vs true values"""
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, s=60, edgecolor="none")
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")
    
    plt.xlabel("True Values", fontsize=14, fontweight="bold")
    plt.ylabel("Predicted Values", fontsize=14, fontweight="bold")
    plt.title(f"{target_name}: Predicted vs True ({model_name})", fontsize=16, fontweight="bold")
    
    metrics_text = f"R² = {r2:.4f}\nMAE = {mae:.4f}\nRMSE = {rmse:.4f}"
    plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    filename = f"{target_name.lower().replace(' ', '_')}_{model_name.lower().replace(' ', '_')}_scatter.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"  Plot saved: {filename}")
    plt.close()


def train_and_evaluate(X, y, target_name, random_state=42):
    """Train with multiple models and ensemble"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=random_state, shuffle=True
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Scale features for neural network
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define base models
    models = {
        "RF": RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_split=3,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        ),
        "LGBM": LGBMRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=50,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1
        ),
        "GB": GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=random_state
        ),
        "MLP": MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.01,
            max_iter=1000,
            early_stopping=True,
            random_state=random_state
        )
    }
    
    results = {}
    
    # Train each model
    for name, model in models.items():
        print(f"\n  Training {name}...")
        
        if name == "MLP":
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            # CV for MLP
            kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
            cv_scores = []
            for train_idx, val_idx in kf.split(X_train_scaled):
                model_cv = MLPRegressor(
                    hidden_layer_sizes=(128, 64, 32),
                    activation='relu',
                    solver='adam',
                    alpha=0.01,
                    max_iter=1000,
                    early_stopping=True,
                    random_state=random_state
                )
                model_cv.fit(X_train_scaled[train_idx], y_train.iloc[train_idx])
                cv_scores.append(r2_score(y_train.iloc[val_idx], 
                                          model_cv.predict(X_train_scaled[val_idx])))
            cv_scores = np.array(cv_scores)
        else:
            # Cross-validation
            kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
            cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"    CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"    Test R²: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        
        results[name] = {
            "model": model,
            "cv_r2": cv_scores.mean(),
            "test_r2": r2,
            "mae": mae,
            "rmse": rmse,
            "y_pred": y_pred,
            "y_test": y_test
        }
        
        plot_predictions(y_test, y_pred, target_name, r2, mae, rmse, name)
    
    # Ensemble (weighted average)
    print(f"\n  Creating Ensemble...")
    
    # Get predictions from all models
    preds = []
    weights = []
    for name, res in results.items():
        if name == "MLP":
            pred = res["model"].predict(X_test_scaled)
        else:
            pred = res["model"].predict(X_test)
        preds.append(pred)
        # Weight by CV R² (normalized)
        weight = max(0, res["cv_r2"])
        weights.append(weight)
    
    # Normalize weights
    weights = np.array(weights)
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.ones(len(weights)) / len(weights)
    
    # Weighted ensemble
    ensemble_pred = np.average(preds, axis=0, weights=weights)
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
    ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
    
    print(f"    Ensemble Weights: {dict(zip(results.keys(), weights))}")
    print(f"    Ensemble Test R²: {ensemble_r2:.4f}")
    print(f"    Ensemble MAE: {ensemble_mae:.4f}")
    print(f"    Ensemble RMSE: {ensemble_rmse:.4f}")
    
    results["Ensemble"] = {
        "test_r2": ensemble_r2,
        "mae": ensemble_mae,
        "rmse": ensemble_rmse,
        "y_pred": ensemble_pred,
        "y_test": y_test,
        "weights": weights
    }
    
    plot_predictions(y_test, ensemble_pred, target_name, ensemble_r2, 
                     ensemble_mae, ensemble_rmse, "Ensemble")
    
    # Find best single model
    best_single = max([(n, r["test_r2"]) for n, r in results.items() if n != "Ensemble"], 
                      key=lambda x: x[1])
    
    print(f"\n  Best Single Model: {best_single[0]} (R² = {best_single[1]:.4f})")
    print(f"  Ensemble R²: {ensemble_r2:.4f}")
    
    if ensemble_r2 > best_single[1]:
        print(f"  -> Ensemble is best!")
        final_r2 = ensemble_r2
    else:
        print(f"  -> {best_single[0]} is best!")
        final_r2 = best_single[1]
    
    return results, final_r2


def main():
    """Main function"""
    print("=" * 60)
    print("Advanced Modeling for Co-enamine Dataset")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    df, descriptors = load_data()
    print(f"Dataset: {df.shape[0]} samples")
    
    # Match descriptors with 8 PCA components
    print("\nMatching descriptors (8 PCA components)...")
    X = match_descriptors(df, descriptors, n_components=8)
    print(f"Features: {X.shape[1]}")
    
    # Predict yield
    print("\n" + "=" * 60)
    print("Predicting YIELD")
    print("=" * 60)
    y_yield = df["yield"]
    yield_results, yield_r2 = train_and_evaluate(X, y_yield, "Yield")
    
    # Predict ee
    print("\n" + "=" * 60)
    print("Predicting ENANTIOMERIC EXCESS (ee)")
    print("=" * 60)
    y_ee = df["ee"]
    ee_results, ee_r2 = train_and_evaluate(X, y_ee, "Enantiomeric Excess")
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    print("\nYield Prediction:")
    for name, res in yield_results.items():
        print(f"  {name:12s}: R² = {res['test_r2']:.4f}")
    
    print("\nEnantiomeric Excess Prediction:")
    for name, res in ee_results.items():
        print(f"  {name:12s}: R² = {res['test_r2']:.4f}")
    
    print(f"\n{'='*60}")
    print(f"BEST RESULTS:")
    print(f"  Yield R²: {yield_r2:.4f}")
    print(f"  ee R²:    {ee_r2:.4f}")
    
    if yield_r2 >= 0.6 and ee_r2 >= 0.6:
        print("\n  ✅ BOTH TARGETS ACHIEVED R² >= 0.6!")
    elif yield_r2 >= 0.6:
        print("\n  ⚠️  Only Yield achieved R² >= 0.6")
    elif ee_r2 >= 0.6:
        print("\n  ⚠️  Only ee achieved R² >= 0.6")
    else:
        print("\n  ❌ Neither target achieved R² >= 0.6")
    
    print("=" * 60)


if __name__ == "__main__":
    main()