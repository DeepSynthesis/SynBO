import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesRegressor
from rdkit import Chem
from rdkit.Chem import Descriptors
import xgboost as xgb
from itertools import product
import os

sns.set_style("whitegrid")


def smiles_to_descriptors(smiles):
    """Generate RDKit molecular descriptors only"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(len(Descriptors._descList))

        desc_list = []
        for _, desc_func in Descriptors._descList:
            try:
                val = desc_func(mol)
                if np.isnan(val) or np.isinf(val):
                    val = 0
                desc_list.append(val)
            except:
                desc_list.append(0)
        return np.array(desc_list)
    except:
        return np.zeros(len(Descriptors._descList))


def load_data():
    """Load main dataset and DFT descriptor files"""
    df = pd.read_csv("Co-enamine.csv")

    # Load DFT descriptors
    dft_files = {
        "amine_smiles": "descriptors/amine_desc_dft.csv",
        "cobalt_smiles": "descriptors/cobalt_desc_dft.csv",
        "oxidant_smiles": "descriptors/oxidant_desc_dft.csv",
        "alkali_smiles": "descriptors/alkali_desc_dft.csv",
        "solvent_smiles": "descriptors/solvent_desc_dft.csv",
    }

    dft_descriptors = {}
    for col, filepath in dft_files.items():
        try:
            dft_df = pd.read_csv(filepath)
            dft_df = dft_df.set_index("SMILES")
            dft_descriptors[col] = dft_df
            print(f"  Loaded DFT descriptors for {col}: {dft_df.shape[1]} features")
        except Exception as e:
            print(f"  Warning: Could not load DFT descriptors for {col}: {e}")
            dft_descriptors[col] = None

    return df, dft_descriptors


def generate_features(df, dft_descriptors):
    """Generate features using RDKit descriptors and DFT descriptors only"""
    feature_list = []
    smiles_cols = ["amine_smiles", "cobalt_smiles", "oxidant_smiles", "alkali_smiles", "solvent_smiles"]

    print("Generating RDKit descriptors and adding DFT descriptors...")

    for smiles_col in smiles_cols:
        print(f"  Processing {smiles_col}...")

        # RDKit descriptors only (no fingerprints)
        rdkit_descs = []
        for smiles in df[smiles_col]:
            desc = smiles_to_descriptors(smiles)
            rdkit_descs.append(desc)

        rdkit_descs = np.array(rdkit_descs)
        desc_cols = [f"{smiles_col}_rdkitdesc_{i}" for i in range(rdkit_descs.shape[1])]
        desc_df = pd.DataFrame(data=rdkit_descs, columns=desc_cols)

        # Add DFT descriptors if available
        if dft_descriptors[smiles_col] is not None:
            dft_df = dft_descriptors[smiles_col]
            dft_features = []
            for smiles in df[smiles_col]:
                if smiles in dft_df.index:
                    dft_features.append(dft_df.loc[smiles].values)
                else:
                    dft_features.append(np.zeros(dft_df.shape[1]))

            dft_features = np.array(dft_features)
            dft_cols = [f"{smiles_col}_dft_{i}" for i in range(dft_features.shape[1])]
            dft_feature_df = pd.DataFrame(data=dft_features, columns=dft_cols)
            combined = pd.concat([desc_df, dft_feature_df], axis=1)
            print(f"    RDKit: {rdkit_descs.shape[1]} features, DFT: {dft_features.shape[1]} features")
        else:
            combined = desc_df
            print(f"    RDKit: {rdkit_descs.shape[1]} features, DFT: 0 features")

        feature_list.append(combined)

    X = pd.concat(feature_list, axis=1)
    return X


def get_unique_reagents(df):
    """Get unique reagents for each column"""
    smiles_cols = ["amine_smiles", "cobalt_smiles", "oxidant_smiles", "alkali_smiles", "solvent_smiles"]

    unique_reagents = {}
    for col in smiles_cols:
        unique_reagents[col] = df[col].unique()
        print(f"  {col}: {len(unique_reagents[col])} unique reagents")

    return unique_reagents


def create_cartesian_product(unique_reagents):
    """Create cartesian product of all reagent combinations"""
    smiles_cols = ["amine_smiles", "cobalt_smiles", "oxidant_smiles", "alkali_smiles", "solvent_smiles"]

    # Get lists of reagents in order
    reagent_lists = [unique_reagents[col] for col in smiles_cols]

    # Create cartesian product
    all_combinations = list(product(*reagent_lists))

    # Create DataFrame
    df_cartesian = pd.DataFrame(all_combinations, columns=smiles_cols)

    print(f"\nTotal combinations: {len(df_cartesian)}")

    return df_cartesian


def train_models(X, y_yield, y_ee):
    """Train yield and ee models with optimized parameters"""

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

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Scale targets for auxiliary features
    scaler_yield = StandardScaler()
    scaler_ee = StandardScaler()
    yield_scaled = scaler_yield.fit_transform(y_yield.values.reshape(-1, 1))
    ee_scaled = scaler_ee.fit_transform(y_ee.values.reshape(-1, 1))

    # Feature selection for yield model
    print("\nTraining Yield model...")
    selector_yield = SelectFromModel(ExtraTreesRegressor(n_estimators=200, random_state=42), max_features=600, threshold=-np.inf)
    X_yield = np.hstack([X_scaled, ee_scaled])
    X_yield_sel = selector_yield.fit_transform(X_yield, y_yield)

    model_yield = xgb.XGBRegressor(
        n_estimators=yield_params["n_estimators"],
        max_depth=yield_params["max_depth"],
        learning_rate=yield_params["learning_rate"],
        subsample=yield_params["subsample"],
        colsample_bytree=yield_params["colsample_bytree"],
        reg_alpha=yield_params["reg_alpha"],
        reg_lambda=yield_params["reg_lambda"],
        min_child_weight=yield_params["min_child_weight"],
        gamma=yield_params["gamma"],
        random_state=42,
        n_jobs=-1,
    )
    model_yield.fit(X_yield_sel, y_yield)
    print(f"  Yield model trained with {X_yield_sel.shape[1]} features")

    # Feature selection for ee model
    print("\nTraining ee model...")
    selector_ee = SelectFromModel(ExtraTreesRegressor(n_estimators=200, random_state=42), max_features=500, threshold=-np.inf)
    X_ee = np.hstack([X_scaled, yield_scaled])
    X_ee_sel = selector_ee.fit_transform(X_ee, y_ee)

    model_ee = xgb.XGBRegressor(
        n_estimators=ee_params["n_estimators"],
        max_depth=ee_params["max_depth"],
        learning_rate=ee_params["learning_rate"],
        subsample=ee_params["subsample"],
        colsample_bytree=ee_params["colsample_bytree"],
        reg_alpha=ee_params["reg_alpha"],
        reg_lambda=ee_params["reg_lambda"],
        min_child_weight=ee_params["min_child_weight"],
        gamma=ee_params["gamma"],
        random_state=42,
        n_jobs=-1,
    )
    model_ee.fit(X_ee_sel, y_ee)
    print(f"  ee model trained with {X_ee_sel.shape[1]} features")

    return {
        "model_yield": model_yield,
        "model_ee": model_ee,
        "selector_yield": selector_yield,
        "selector_ee": selector_ee,
        "scaler": scaler,
        "scaler_yield": scaler_yield,
        "scaler_ee": scaler_ee,
    }


def predict_on_cartesian(df_cartesian, models, dft_descriptors):
    """Predict yield and ee on cartesian product of reagents"""

    # Generate features for cartesian product
    print("\nGenerating features for cartesian product...")
    X_cartesian = generate_features(df_cartesian, dft_descriptors)
    X_cartesian_scaled = models["scaler"].transform(X_cartesian)

    # For prediction, we need to use auxiliary features
    # We'll use mean values as initial estimates
    print("Predicting...")

    # First pass: predict with mean auxiliary values
    mean_yield = 50.0  # Assume 50% yield initially
    mean_ee = 50.0  # Assume 50% ee initially

    # Iterative prediction
    n_iterations = 3
    for iteration in range(n_iterations):
        # Prepare features with auxiliary values
        yield_aux = np.full((len(df_cartesian), 1), mean_yield)
        ee_aux = np.full((len(df_cartesian), 1), mean_ee)

        yield_aux_scaled = models["scaler_yield"].transform(yield_aux)
        ee_aux_scaled = models["scaler_ee"].transform(ee_aux)

        # Predict yield
        X_yield = np.hstack([X_cartesian_scaled, ee_aux_scaled])
        X_yield_sel = models["selector_yield"].transform(X_yield)
        yield_pred = models["model_yield"].predict(X_yield_sel)
        yield_pred = np.clip(yield_pred, 0, 100)

        # Predict ee
        X_ee = np.hstack([X_cartesian_scaled, yield_aux_scaled])
        X_ee_sel = models["selector_ee"].transform(X_ee)
        ee_pred = models["model_ee"].predict(X_ee_sel)
        ee_pred = np.clip(ee_pred, 0, 100)

        # Update for next iteration
        mean_yield = yield_pred.mean()
        mean_ee = ee_pred.mean()

        print(f"  Iteration {iteration + 1}: mean_yield={mean_yield:.2f}, mean_ee={mean_ee:.2f}")

    return yield_pred, ee_pred


def save_results(df_cartesian, yield_pred, ee_pred, filename="cartesian_predictions.npy"):
    """Save results to npy file"""
    # Create results array
    results = np.column_stack([yield_pred, ee_pred])

    # Save as npy
    np.save(filename, results)
    print(f"\nResults saved to {filename}")

    # Also save as CSV for inspection
    df_results = df_cartesian.copy()
    df_results["yield_pred"] = yield_pred
    df_results["ee_pred"] = ee_pred

    csv_filename = filename.replace(".npy", ".csv")
    df_results.to_csv(csv_filename, index=False)
    print(f"Results also saved to {csv_filename}")

    return df_results


def plot_density_map(df_results, filename="ee_yield_density.png"):
    """Plot density map of ee vs yield"""
    plt.figure(figsize=(10, 8))

    # Create 2D histogram / density plot
    x = df_results["ee_pred"]
    y = df_results["yield_pred"]

    # Create hexbin plot for density
    hb = plt.hexbin(x, y, gridsize=30, cmap="YlOrRd", mincnt=1)

    # Add colorbar
    cb = plt.colorbar(hb)
    cb.set_label("Count", fontsize=12)

    # Labels and title
    plt.xlabel("Predicted ee (%)", fontsize=14, fontweight="bold")
    plt.ylabel("Predicted Yield (%)", fontsize=14, fontweight="bold")
    plt.title("Density Map: ee vs Yield (Cartesian Product)", fontsize=16, fontweight="bold")

    # Set limits
    plt.xlim(0, 100)
    plt.ylim(0, 100)

    # Add grid
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Density plot saved to {filename}")
    plt.close()


def plot_scatter(df_results, filename="ee_yield_scatter.png"):
    """Plot scatter plot of ee vs yield colored by density"""
    plt.figure(figsize=(10, 8))

    # Calculate point density
    from scipy.stats import gaussian_kde

    x = df_results["ee_pred"].values
    y = df_results["yield_pred"].values

    # Create density coloring
    xy = np.vstack([x, y])
    try:
        z = gaussian_kde(xy)(xy)

        # Sort points by density for better visualization
        idx = z.argsort()
        x_sorted = x[idx]
        y_sorted = y[idx]
        z_sorted = z[idx]

        plt.scatter(x_sorted, y_sorted, c=z_sorted, s=20, cmap="viridis", alpha=0.6)
        plt.colorbar(label="Density")
    except:
        # Fallback to simple scatter if density calculation fails
        plt.scatter(x, y, s=20, alpha=0.6, c="blue")

    plt.xlabel("Predicted ee (%)", fontsize=14, fontweight="bold")
    plt.ylabel("Predicted Yield (%)", fontsize=14, fontweight="bold")
    plt.title("Scatter Plot: ee vs Yield (Cartesian Product)", fontsize=16, fontweight="bold")
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Scatter plot saved to {filename}")
    plt.close()


def main():
    """Main function"""
    print("=" * 70)
    print("Mapping Results - Cartesian Product Prediction")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    df, dft_descriptors = load_data()
    print(f"\nOriginal dataset: {df.shape[0]} samples")

    # Get unique reagents
    print("\nExtracting unique reagents...")
    unique_reagents = get_unique_reagents(df)

    # Generate features for original data
    print("\nGenerating features for original data...")
    X = generate_features(df, dft_descriptors)
    print(f"\nTotal features: {X.shape[1]}")

    y_yield = df["yield"]
    y_ee = df["ee"]

    # Train models
    print("\n" + "=" * 70)
    print("Training Models")
    print("=" * 70)
    models = train_models(X, y_yield, y_ee)

    # Create cartesian product
    print("\n" + "=" * 70)
    print("Creating Cartesian Product")
    print("=" * 70)
    df_cartesian = create_cartesian_product(unique_reagents)

    # Predict on cartesian product
    print("\n" + "=" * 70)
    print("Predicting on Cartesian Product")
    print("=" * 70)
    yield_pred, ee_pred = predict_on_cartesian(df_cartesian, models, dft_descriptors)

    # Save results
    print("\n" + "=" * 70)
    print("Saving Results")
    print("=" * 70)
    df_results = save_results(df_cartesian, yield_pred, ee_pred)

    # Print statistics
    print("\n" + "=" * 70)
    print("Prediction Statistics")
    print("=" * 70)
    print(f"\nYield predictions:")
    print(f"  Min: {yield_pred.min():.2f}")
    print(f"  Max: {yield_pred.max():.2f}")
    print(f"  Mean: {yield_pred.mean():.2f}")
    print(f"  Std: {yield_pred.std():.2f}")

    print(f"\nee predictions:")
    print(f"  Min: {ee_pred.min():.2f}")
    print(f"  Max: {ee_pred.max():.2f}")
    print(f"  Mean: {ee_pred.mean():.2f}")
    print(f"  Std: {ee_pred.std():.2f}")

    # Plot density map
    print("\n" + "=" * 70)
    print("Generating Plots")
    print("=" * 70)
    plot_density_map(df_results)
    plot_scatter(df_results)

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
