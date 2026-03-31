import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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


def match_descriptors_improved(df, descriptors, n_components=5):
    """Match descriptors and apply PCA with more components"""
    feature_list = []
    feature_names = []
    
    for smiles_col, desc_df in descriptors.items():
        # Create mapping from SMILES to descriptors
        desc_dict = {}
        for idx, row in desc_df.iterrows():
            smiles = row["SMILES"]
            desc_cols = [col for col in desc_df.columns if col != "SMILES"]
            desc_dict[smiles] = row[desc_cols].values
        
        # Match descriptors
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
        
        print(f"{smiles_col}: Reduced from {matched_desc.shape[1]} to {n_comp} dimensions "
              f"(variance explained: {pca.explained_variance_ratio_.sum():.4f})")
    
    X = pd.concat(feature_list, axis=1)
    return X


def add_interaction_features(X):
    """Add interaction features between different categories"""
    X_new = X.copy()
    cols = X.columns.tolist()
    
    # Add squared features for each component
    for col in cols:
        X_new[f"{col}_sq"] = X[col] ** 2
    
    # Add interaction features between different categories
    categories = set([c.split('_PC')[0] for c in cols])
    cat_cols = {cat: [c for c in cols if c.startswith(cat)] for cat in categories}
    
    cat_list = list(categories)
    for i in range(len(cat_list)):
        for j in range(i+1, len(cat_list)):
            cat1, cat2 = cat_list[i], cat_list[j]
            for col1 in cat_cols[cat1][:2]:  # Only first 2 PCs
                for col2 in cat_cols[cat2][:2]:
                    X_new[f"{col1}_x_{col2}"] = X[col1] * X[col2]
    
    return X_new


def remove_outliers(X, y, threshold=3):
    """Remove outliers using Z-score"""
    z_scores = np.abs(stats.zscore(y))
    mask = z_scores < threshold
    return X[mask], y[mask]


def plot_predictions(y_true, y_pred, target_name, r2, mae, rmse, cv_r2=None):
    """Plot predicted vs true values"""
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, s=60, edgecolor="none")
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")
    
    plt.xlabel("True Values", fontsize=14, fontweight="bold")
    plt.ylabel("Predicted Values", fontsize=14, fontweight="bold")
    plt.title(f"{target_name}: Predicted vs True Values", fontsize=16, fontweight="bold")
    
    metrics_text = f"R² = {r2:.4f}"
    if cv_r2 is not None:
        metrics_text += f"\nCV R² = {cv_r2:.4f}"
    metrics_text += f"\nMAE = {mae:.4f}\nRMSE = {rmse:.4f}"
    
    plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    filename = f"{target_name.lower().replace(' ', '_')}_prediction_scatter.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"\nScatter plot saved as: {filename}")
    plt.close()


def train_and_evaluate_improved(X, y, target_name="", random_state=42):
    """Train with cross-validation and ensemble"""
    # Remove outliers
    X_clean, y_clean = remove_outliers(X, y, threshold=3)
    print(f"Removed {len(y) - len(y_clean)} outliers, remaining: {len(y_clean)}")
    
    # Add interaction features
    X_features = add_interaction_features(X_clean)
    print(f"Feature count after adding interactions: {X_features.shape[1]}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)
    X_scaled = pd.DataFrame(X_scaled, columns=X_features.columns)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_clean, test_size=0.2, random_state=random_state, shuffle=True
    )
    
    # Optimized LightGBM parameters
    best_params = {
        'n_estimators': 300,
        'max_depth': 8,
        'learning_rate': 0.05,
        'num_leaves': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'min_child_samples': 10,
        'random_state': random_state,
        'n_jobs': -1,
        'verbose': -1,
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse'
    }
    
    # Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    model = LGBMRegressor(**best_params)
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
    print(f"\nCross-validation R² scores: {cv_scores}")
    print(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Train final model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\nTest Set Results:")
    print(f"  R² = {r2:.4f}")
    print(f"  MAE = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    
    # Plot
    plot_predictions(y_test, y_pred, target_name, r2, mae, rmse, cv_r2=cv_scores.mean())
    
    return {
        "model": model,
        "scaler": scaler,
        "y_test": y_test,
        "y_pred": y_pred,
        "r2": r2,
        "cv_r2": cv_scores.mean(),
        "mae": mae,
        "rmse": rmse
    }


def main():
    print("=" * 60)
    print("Improved LightGBM Modeling for Co-enamine Dataset")
    print("=" * 60)
    
    print("\nLoading data...")
    df, descriptors = load_data()
    print(f"Main dataset shape: {df.shape}")
    
    print("\nMatching descriptors with PCA (5 components)...")
    X = match_descriptors_improved(df, descriptors, n_components=5)
    print(f"Feature matrix shape: {X.shape}")
    
    # Predict yield
    print("\n" + "=" * 60)
    print("Predicting YIELD")
    print("=" * 60)
    y_yield = df["yield"]
    yield_results = train_and_evaluate_improved(X, y_yield, target_name="Yield", random_state=42)
    
    # Predict ee
    print("\n" + "=" * 60)
    print("Predicting ENANTIOMERIC EXCESS (ee)")
    print("=" * 60)
    y_ee = df["ee"]
    ee_results = train_and_evaluate_improved(X, y_ee, target_name="Enantiomeric Excess", random_state=42)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\nYield Prediction Results:")
    print(f"  Test R²: {yield_results['r2']:.4f}")
    print(f"  CV R²: {yield_results['cv_r2']:.4f}")
    print(f"  MAE: {yield_results['mae']:.4f}")
    print(f"  RMSE: {yield_results['rmse']:.4f}")
    
    print("\nEnantiomeric Excess Prediction Results:")
    print(f"  Test R²: {ee_results['r2']:.4f}")
    print(f"  CV R²: {ee_results['cv_r2']:.4f}")
    print(f"  MAE: {ee_results['mae']:.4f}")
    print(f"  RMSE: {ee_results['rmse']:.4f}")
    
    print("\n" + "=" * 60)
    print("Modeling completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()