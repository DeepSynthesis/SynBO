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
import time

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

    dft_files = {
        "amine": "descriptors/amine_desc_dft.csv",
        "cobalt": "descriptors/cobalt_desc_dft.csv",
        "oxidant": "descriptors/oxidant_desc_dft.csv",
        "alkali": "descriptors/alkali_desc_dft.csv",
        "solvent": "descriptors/solvent_desc_dft.csv",
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


def get_unique_reagents(df):
    """Get unique reagents for each column"""
    smiles_cols = ["amine", "cobalt", "oxidant", "alkali", "solvent"]
    unique_reagents = {}
    for col in smiles_cols:
        unique_reagents[col] = df[col].unique()
        print(f"  {col}: {len(unique_reagents[col])} unique reagents")
    return unique_reagents


# ================== 核心优化 1：预计算分子特征 ==================
def precompute_features(unique_reagents, dft_descriptors):
    """Precompute RDKit and DFT features for ONLY the unique molecules (extremely fast)"""
    print("\nPrecomputing features for unique molecules...")
    precomputed_dict = {}

    for col, smiles_list in unique_reagents.items():
        dft_df = dft_descriptors[col]
        dft_dim = dft_df.shape[1] if dft_df is not None else 0
        col_features = {}

        for smiles in smiles_list:
            # 1. RDKit features
            rdkit_desc = smiles_to_descriptors(smiles)

            # 2. DFT features
            if dft_df is not None and smiles in dft_df.index:
                dft_desc = dft_df.loc[smiles].values
            else:
                dft_desc = np.zeros(dft_dim)

            # Combine
            col_features[smiles] = np.concatenate([rdkit_desc, dft_desc])

        precomputed_dict[col] = col_features
    return precomputed_dict


# ================== 核心优化 2：向量化拼接特征 ==================
def assemble_features_fast(df, precomputed_dict):
    """Assemble feature matrix using precomputed dictionaries (orders of magnitude faster)"""
    smiles_cols = ["amine", "cobalt", "oxidant", "alkali", "solvent"]

    parts = []
    for col in smiles_cols:
        # Fast list comprehension to get array for this column
        arr = np.array([precomputed_dict[col][smiles] for smiles in df[col]])
        parts.append(arr)

    # Stack all columns horizontally
    X = np.hstack(parts)
    return X


def create_cartesian_product(unique_reagents):
    """Create cartesian product of all reagent combinations"""
    smiles_cols = ["amine", "cobalt", "oxidant", "alkali", "solvent"]
    reagent_lists = [unique_reagents[col] for col in smiles_cols]
    all_combinations = list(product(*reagent_lists))
    df_cartesian = pd.DataFrame(all_combinations, columns=smiles_cols)
    print(f"\nTotal combinations: {len(df_cartesian)}")
    return df_cartesian


def train_models(X, y_yield, y_ee):
    """Train yield and ee models with optimized parameters"""
    yield_params = {
        "n_estimators": 2650,
        "max_depth": 14,
        "learning_rate": 0.0013881177423179518,
        "subsample": 0.8832263299249901,
        "colsample_bytree": 0.7903362550905764,
        "reg_alpha": 1.3089861430282985,
        "reg_lambda": 1.5737502114895576,
        "min_child_weight": 8,
        "gamma": 0.7509204269440869,
        "n_features": 200,
    }
    ee_params = {
        "n_estimators": 2450,
        "max_depth": 11,
        "learning_rate": 0.0013346474415975344,
        "subsample": 0.933922897579847,
        "colsample_bytree": 0.869612716429276,
        "reg_alpha": 1.5160690367861887,
        "reg_lambda": 0.9174612897194362,
        "min_child_weight": 4,
        "gamma": 0.606792312408108,
        "n_features": 200,
    }

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scaler_yield = StandardScaler()
    scaler_ee = StandardScaler()
    yield_scaled = scaler_yield.fit_transform(y_yield.values.reshape(-1, 1))
    ee_scaled = scaler_ee.fit_transform(y_ee.values.reshape(-1, 1))

    print("\nTraining Yield model...")
    selector_yield = SelectFromModel(ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1), max_features=600, threshold=-np.inf)
    X_yield = np.hstack([X_scaled, ee_scaled])
    X_yield_sel = selector_yield.fit_transform(X_yield, y_yield)

    model_yield = xgb.XGBRegressor(**yield_params, random_state=42, n_jobs=-1)
    model_yield.fit(X_yield_sel, y_yield)
    print(f"  Yield model trained with {X_yield_sel.shape[1]} features")

    print("\nTraining ee model...")
    selector_ee = SelectFromModel(ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1), max_features=500, threshold=-np.inf)
    X_ee = np.hstack([X_scaled, yield_scaled])
    X_ee_sel = selector_ee.fit_transform(X_ee, y_ee)

    model_ee = xgb.XGBRegressor(**ee_params, random_state=42, n_jobs=-1)
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


def predict_on_cartesian(df_cartesian, models, precomputed_dict):
    """Predict yield and ee on cartesian product of reagents"""
    print("\nAssembling features for cartesian product (fast)...")
    start_time = time.time()

    # Use fast assembly function
    X_cartesian = assemble_features_fast(df_cartesian, precomputed_dict)
    X_cartesian_scaled = models["scaler"].transform(X_cartesian)

    print(f"Feature assembly took {time.time() - start_time:.2f} seconds")

    print("Predicting...")
    mean_yield, mean_ee = 50.0, 50.0

    for iteration in range(3):
        yield_aux = np.full((len(df_cartesian), 1), mean_yield)
        ee_aux = np.full((len(df_cartesian), 1), mean_ee)

        yield_aux_scaled = models["scaler_yield"].transform(yield_aux)
        ee_aux_scaled = models["scaler_ee"].transform(ee_aux)

        # Predict yield
        X_yield = np.hstack([X_cartesian_scaled, ee_aux_scaled])
        X_yield_sel = models["selector_yield"].transform(X_yield)
        yield_pred = np.clip(models["model_yield"].predict(X_yield_sel), 0, 100)

        # Predict ee
        X_ee = np.hstack([X_cartesian_scaled, yield_aux_scaled])
        X_ee_sel = models["selector_ee"].transform(X_ee)
        ee_pred = np.clip(models["model_ee"].predict(X_ee_sel), 0, 100)

        mean_yield, mean_ee = yield_pred.mean(), ee_pred.mean()
        print(f"  Iteration {iteration + 1}: mean_yield={mean_yield:.2f}, mean_ee={mean_ee:.2f}")

    return yield_pred, ee_pred


def save_results(df_cartesian, yield_pred, ee_pred, filename="cartesian_predictions.npy"):
    results = np.column_stack([yield_pred, ee_pred])
    np.save(filename, results)

    df_results = df_cartesian.copy()
    df_results["yield_pred"] = yield_pred
    df_results["ee_pred"] = ee_pred
    df_results.to_csv(filename.replace(".npy", ".csv"), index=False)
    print(f"\nResults saved to {filename} and CSV")
    return df_results


def plot_density_map(df_results, filename="ee_yield_density.png"):
    plt.figure(figsize=(10, 8))
    x = df_results["ee_pred"]
    y = df_results["yield_pred"]
    hb = plt.hexbin(x, y, gridsize=30, cmap="YlOrRd", mincnt=1)
    cb = plt.colorbar(hb)
    cb.set_label("Count", fontsize=12)
    plt.xlabel("Predicted ee (%)", fontsize=14, fontweight="bold")
    plt.ylabel("Predicted Yield (%)", fontsize=14, fontweight="bold")
    plt.title("Density Map: ee vs Yield", fontsize=16, fontweight="bold")
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# ================== 核心优化 3：极速散点密度图 ==================
def plot_scatter(df_results, filename="ee_yield_scatter.png"):
    """Plot scatter plot of ee vs yield colored by density using fast 2D histogram"""
    plt.figure(figsize=(10, 8))

    x = df_results["ee_pred"].values
    y = df_results["yield_pred"].values

    # 使用 np.histogram2d 替代极慢的 scipy gaussian_kde
    bins = 50
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins)

    # 映射密度值到每个点
    x_idx = np.clip(np.digitize(x, xedges) - 1, 0, bins - 1)
    y_idx = np.clip(np.digitize(y, yedges) - 1, 0, bins - 1)
    z = hist[x_idx, y_idx]

    # 按密度排序，确保高密度点在最上层
    idx = z.argsort()
    x_sorted, y_sorted, z_sorted = x[idx], y[idx], z[idx]

    plt.scatter(x_sorted, y_sorted, c=z_sorted, s=20, cmap="viridis", alpha=0.6, edgecolor="none")
    plt.colorbar(label="Density")

    plt.xlabel("Predicted ee (%)", fontsize=14, fontweight="bold")
    plt.ylabel("Predicted Yield (%)", fontsize=14, fontweight="bold")
    plt.title("Scatter Plot: ee vs Yield", fontsize=16, fontweight="bold")
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def main():
    print("=" * 70)
    print("Mapping Results - Accelerated Cartesian Product Prediction")
    print("=" * 70)

    # 1. Load data & get uniques
    df, dft_descriptors = load_data()
    unique_reagents = get_unique_reagents(df)

    # 2. Precompute features ONLY for unique molecules (Fast!)
    precomputed_dict = precompute_features(unique_reagents, dft_descriptors)

    # 3. Assemble features for original dataset
    X = assemble_features_fast(df, precomputed_dict)
    y_yield, y_ee = df["yield"], df["ee"]

    # 4. Train models
    models = train_models(X, y_yield, y_ee)

    # 5. Create Cartesian & Predict
    df_cartesian = create_cartesian_product(unique_reagents)
    yield_pred, ee_pred = predict_on_cartesian(df_cartesian, models, precomputed_dict)

    # 6. Save & Plot
    df_results = save_results(df_cartesian, yield_pred, ee_pred)
    plot_density_map(df_results)
    plot_scatter(df_results)

    print("\nDone!")


if __name__ == "__main__":
    main()
