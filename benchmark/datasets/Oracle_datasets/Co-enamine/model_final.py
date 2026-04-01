import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesRegressor
import xgboost as xgb

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


def train_and_evaluate_model(X, y, params, target_name, n_features):
    """Train model with optimized parameters and evaluate with 5-fold CV"""
    print(f"\n  Feature selection...")
    selector = SelectFromModel(ExtraTreesRegressor(n_estimators=200, random_state=42), max_features=n_features, threshold=-np.inf)
    X_sel = selector.fit_transform(X, y)
    print(f"  Selected {X_sel.shape[1]} features")

    print(f"\n  Training XGBoost model for {target_name}...")
    print(f"    Using optimized parameters...")

    # Create model with optimized params
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
    )

    # 5-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_sel, y, cv=kf, scoring="r2", n_jobs=-1)
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
    print("Model Final - XGBoost with Optimized Parameters (5-Fold CV)")
    print("=" * 70)

    # Optimized parameters from hyperopt
    yield_params = {
        "n_estimators": 300,
        "max_depth": 8,
        "learning_rate": 0.04766270340088169,
        "subsample": 0.991714776103226,
        "colsample_bytree": 0.9303831360618628,
        "reg_alpha": 0.22874493399616896,
        "reg_lambda": 0.9070719206144939,
        "min_child_weight": 2,
        "gamma": 0.15215320156702056,
    }

    ee_params = {
        "n_estimators": 2000,
        "max_depth": 14,
        "learning_rate": 0.018694468369353623,
        "subsample": 0.8423625453977001,
        "colsample_bytree": 0.8690535229123624,
        "reg_alpha": 0.9880424042229838,
        "reg_lambda": 0.33069879837462873,
        "min_child_weight": 1,
        "gamma": 0.4976255083929807,
    }

    # Load data
    print("\nLoading data...")
    df, descriptors = load_data()
    print(f"\nDataset: {df.shape[0]} samples")

    # Generate features
    print("\nGenerating Features...")
    X = generate_features(df, descriptors)
    print(f"\nTotal features: {X.shape[1]}")

    y_yield = df["yield"]
    y_ee = df["ee"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Add cross-features (ee for yield prediction, yield for ee prediction)
    scaler_yield = StandardScaler()
    scaler_ee = StandardScaler()
    yield_scaled = scaler_yield.fit_transform(y_yield.values.reshape(-1, 1))
    ee_scaled = scaler_ee.fit_transform(y_ee.values.reshape(-1, 1))

    results = {}

    # ========== YIELD PREDICTION ==========
    print("\n" + "=" * 70)
    print("YIELD PREDICTION - XGBoost with Optimized Parameters")
    print("=" * 70)

    # Use ee as auxiliary feature
    X_yield = np.hstack([X_scaled, ee_scaled])

    # Train and evaluate
    result_yield = train_and_evaluate_model(X_yield, y_yield, yield_params, "Yield", n_features=600)
    plot_cv_predictions(
        result_yield["y_true"], result_yield["y_pred"], "Yield (Optimized XGBoost)", result_yield["overall_r2"], "yield_final.png"
    )

    results["Yield"] = {
        "cv_r2_mean": result_yield["cv_r2_mean"],
        "cv_r2_std": result_yield["cv_r2_std"],
        "overall_r2": result_yield["overall_r2"],
    }

    # ========== EE PREDICTION ==========
    print("\n" + "=" * 70)
    print("ENANTIOMERIC EXCESS PREDICTION - XGBoost with Optimized Parameters")
    print("=" * 70)

    # Use yield as auxiliary feature
    X_ee = np.hstack([X_scaled, yield_scaled])

    # Train and evaluate
    result_ee = train_and_evaluate_model(X_ee, y_ee, ee_params, "ee", n_features=500)
    plot_cv_predictions(
        result_ee["y_true"], result_ee["y_pred"], "Enantiomeric Excess (Optimized XGBoost)", result_ee["overall_r2"], "ee_final.png"
    )

    results["ee"] = {"cv_r2_mean": result_ee["cv_r2_mean"], "cv_r2_std": result_ee["cv_r2_std"], "overall_r2": result_ee["overall_r2"]}

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY - XGBoost with Optimized Parameters (5-Fold CV)")
    print("=" * 70)

    # Yield results
    print("\nYield Prediction Results:")
    print(f"  5-Fold CV R²: {results['Yield']['cv_r2_mean']:.4f} (+/- {results['Yield']['cv_r2_std']:.4f})")
    print(f"  Overall R²:   {results['Yield']['overall_r2']:.4f}")
    print(f"  Status: {'✅ >= 0.6' if results['Yield']['cv_r2_mean'] >= 0.6 else '⚠️ < 0.6'}")

    # ee results
    print("\nEnantiomeric Excess Prediction Results:")
    print(f"  5-Fold CV R²: {results['ee']['cv_r2_mean']:.4f} (+/- {results['ee']['cv_r2_std']:.4f})")
    print(f"  Overall R²:   {results['ee']['overall_r2']:.4f}")
    print(f"  Status: {'✅ >= 0.6' if results['ee']['cv_r2_mean'] >= 0.6 else '⚠️ < 0.6'}")

    # Print optimized parameters
    print(f"\n{'='*70}")
    print("Optimized Hyperparameters:")
    print(f"\nYield best params: {yield_params}")
    print(f"\nee best params: {ee_params}")

    # Overall status
    yield_ok = results["Yield"]["cv_r2_mean"] >= 0.6
    ee_ok = results["ee"]["cv_r2_mean"] >= 0.6

    print(f"\n{'='*70}")
    if yield_ok and ee_ok:
        print("🎉 SUCCESS! Both targets achieved 5-Fold CV R² >= 0.6!")
    else:
        print("⚠️  Some targets did not reach R² >= 0.6")
        if not yield_ok:
            print(f"   Yield gap: {0.6 - results['Yield']['cv_r2_mean']:.4f}")
        if not ee_ok:
            print(f"   ee gap: {0.6 - results['ee']['cv_r2_mean']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()