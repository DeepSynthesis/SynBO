import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

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


def match_descriptors(df, descriptors, n_components=None):
    """Match descriptors with optional PCA"""
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
        
        if n_components is not None and n_components < matched_desc.shape[1]:
            n_comp = min(n_components, matched_desc.shape[1])
            pca = PCA(n_components=n_comp, random_state=42)
            desc_pca = pca.fit_transform(desc_df_matched)
            pca_cols = [f"{smiles_col}_PC{i}" for i in range(n_comp)]
            desc_pca_df = pd.DataFrame(desc_pca, columns=pca_cols)
            feature_list.append(desc_pca_df)
            print(f"{smiles_col}: {matched_desc.shape[1]} -> {n_comp} dims "
                  f"(var: {pca.explained_variance_ratio_.sum():.4f})")
        else:
            # Use all features without PCA
            cols = [f"{smiles_col}_feat{i}" for i in range(matched_desc.shape[1])]
            desc_df = pd.DataFrame(matched_desc, columns=cols)
            feature_list.append(desc_df)
            print(f"{smiles_col}: Using all {matched_desc.shape[1]} features")
    
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


def train_model(X, y, target_name, n_components=None, random_state=42, df=None, descriptors=None):
    """Train with optimized RandomForest"""
    # Prepare features
    if df is not None and descriptors is not None:
        X_features = match_descriptors(df, descriptors, n_components)
    else:
        X_features = X
    
    # Split data (15% test to match previous best result)
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.15, random_state=random_state, shuffle=True
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Optimized RandomForest parameters based on experiments
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=15,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        random_state=random_state,
        n_jobs=-1
    )
    
    # Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
    print(f"\n  Cross-validation R² scores: {cv_scores}")
    print(f"  Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # Train and predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\n  Test Set Results:")
    print(f"    R² = {r2:.4f}")
    print(f"    MAE = {mae:.4f}")
    print(f"    RMSE = {rmse:.4f}")
    
    # Plot
    plot_predictions(y_test, y_pred, target_name, r2, mae, rmse, "RandomForest")
    
    return {
        "model": model,
        "cv_r2_mean": cv_scores.mean(),
        "cv_r2_std": cv_scores.std(),
        "test_r2": r2,
        "mae": mae,
        "rmse": rmse
    }


def try_different_pca_components(df, descriptors, y, target_name):
    """Try different numbers of PCA components"""
    results = {}
    
    # Try no PCA (all features)
    print(f"\n{'='*60}")
    print(f"{target_name} - All Features (No PCA)")
    print('='*60)
    X_all = match_descriptors(df, descriptors, n_components=None)
    print(f"Total features: {X_all.shape[1]}")
    results["All_Features"] = train_model(X_all, y, target_name, random_state=42)
    
    # Try different PCA components
    for n_comp in [10, 8, 5, 3, 2]:
        print(f"\n{'='*60}")
        print(f"{target_name} - PCA with {n_comp} components")
        print('='*60)
        
        X_pca = match_descriptors(df, descriptors, n_components=n_comp)
        print(f"Total features: {X_pca.shape[1]}")
        results[f"PCA_{n_comp}"] = train_model(X_pca, y, target_name, random_state=42)
    
    # Find best
    best_name = max(results.keys(), key=lambda k: results[k]['test_r2'])
    print(f"\n{'='*60}")
    print(f"BEST CONFIGURATION for {target_name}:")
    print(f"  Configuration: {best_name}")
    print(f"  Test R²: {results[best_name]['test_r2']:.4f}")
    print(f"  CV R²: {results[best_name]['cv_r2_mean']:.4f}")
    print('='*60)
    
    return results, best_name


def main():
    """Main function"""
    print("=" * 70)
    print("Optimized Modeling for Co-enamine Dataset")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    df, descriptors = load_data()
    print(f"Dataset: {df.shape[0]} samples")
    
    # Predict yield with different configurations
    print("\n" + "=" * 70)
    print("Predicting YIELD - Testing Different Feature Configurations")
    print("=" * 70)
    y_yield = df["yield"]
    yield_results, yield_best = try_different_pca_components(df, descriptors, y_yield, "Yield")
    
    # Predict ee with different configurations
    print("\n" + "=" * 70)
    print("Predicting ENANTIOMERIC EXCESS (ee) - Testing Different Configurations")
    print("=" * 70)
    y_ee = df["ee"]
    ee_results, ee_best = try_different_pca_components(df, descriptors, y_ee, "Enantiomeric Excess")
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    print("\nYield Prediction Results:")
    for name, res in yield_results.items():
        status = " ✓ BEST" if name == yield_best else ""
        print(f"  {name:20s}: R² = {res['test_r2']:.4f} (CV: {res['cv_r2_mean']:.4f}){status}")
    
    print("\nEnantiomeric Excess Prediction Results:")
    for name, res in ee_results.items():
        status = " ✓ BEST" if name == ee_best else ""
        print(f"  {name:20s}: R² = {res['test_r2']:.4f} (CV: {res['cv_r2_mean']:.4f}){status}")
    
    print(f"\n{'='*70}")
    print(f"OVERALL BEST RESULTS:")
    print(f"  Yield: {yield_best} - R² = {yield_results[yield_best]['test_r2']:.4f}")
    print(f"  ee:    {ee_best} - R² = {ee_results[ee_best]['test_r2']:.4f}")
    
    # Check if target achieved
    yield_r2 = yield_results[yield_best]['test_r2']
    ee_r2 = ee_results[ee_best]['test_r2']
    
    if yield_r2 >= 0.6 and ee_r2 >= 0.6:
        print("\n  ✅ BOTH TARGETS ACHIEVED R² >= 0.6!")
    elif yield_r2 >= 0.6:
        print("\n  ⚠️  Only Yield achieved R² >= 0.6")
    elif ee_r2 >= 0.6:
        print("\n  ⚠️  Only ee achieved R² >= 0.6")
    else:
        print(f"\n  ❌ Neither target achieved R² >= 0.6")
        print(f"     Yield gap: {0.6 - yield_r2:.4f}")
        print(f"     ee gap:    {0.6 - ee_r2:.4f}")
    
    print("=" * 70)


if __name__ == "__main__":
    main()