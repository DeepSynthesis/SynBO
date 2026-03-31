import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import MACCSkeys, rdMolDescriptors

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")


def smiles_to_fingerprints(smiles, radius=2, n_bits=2048):
    """Convert SMILES to Morgan fingerprints"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(n_bits)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        return np.array(fp)
    except:
        return np.zeros(n_bits)


def smiles_to_maccs(smiles):
    """Convert SMILES to MACCS keys"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(167)
        fp = MACCSkeys.GenMACCSKeys(mol)
        return np.array(fp)
    except:
        return np.zeros(167)


def smiles_to_rdkit_fp(smiles, n_bits=2048):
    """Convert SMILES to RDKit topological fingerprints"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(n_bits)
        fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=n_bits)
        return np.array(fp)
    except:
        return np.zeros(n_bits)


def smiles_to_descriptors(smiles):
    """Generate RDKit molecular descriptors"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(len(Descriptors._descList))
        
        desc_list = []
        for desc_name, desc_func in Descriptors._descList:
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
    """Load main dataset and descriptor files"""
    df = pd.read_csv("Co-enamine.csv")
    return df


def generate_enhanced_features(df):
    """Generate enhanced features using RDKit fingerprints"""
    feature_list = []
    smiles_cols = ['amine_smiles_ion', 'amine_smiles_anion', 'cobalt_smiles', 
                   'oxidant_smiles', 'alkali_smiles', 'solvent_smiles']
    
    print("Generating molecular fingerprints...")
    
    for smiles_col in smiles_cols:
        print(f"  Processing {smiles_col}...")
        
        morgan_fps = []
        maccs_fps = []
        rdkit_fps = []
        descriptors = []
        
        for smiles in df[smiles_col]:
            # Morgan fingerprints
            morgan = smiles_to_fingerprints(smiles, radius=2, n_bits=2048)
            morgan_fps.append(morgan)
            
            # MACCS keys
            maccs = smiles_to_maccs(smiles)
            maccs_fps.append(maccs)
            
            # RDKit topological fingerprints
            rdkit = smiles_to_rdkit_fp(smiles, n_bits=2048)
            rdkit_fps.append(rdkit)
            
            # Molecular descriptors
            desc = smiles_to_descriptors(smiles)
            descriptors.append(desc)
        
        morgan_fps = np.array(morgan_fps)
        maccs_fps = np.array(maccs_fps)
        rdkit_fps = np.array(rdkit_fps)
        descriptors = np.array(descriptors)
        
        # Apply PCA to reduce dimensionality
        n_comp_morgan = min(100, morgan_fps.shape[1])
        n_comp_rdkit = min(100, rdkit_fps.shape[1])
        n_comp_desc = min(50, descriptors.shape[1])
        
        pca_morgan = PCA(n_components=n_comp_morgan, random_state=42)
        pca_rdkit = PCA(n_components=n_comp_rdkit, random_state=42)
        pca_desc = PCA(n_components=n_comp_desc, random_state=42)
        
        morgan_reduced = pca_morgan.fit_transform(morgan_fps)
        rdkit_reduced = pca_rdkit.fit_transform(rdkit_fps)
        desc_reduced = pca_desc.fit_transform(descriptors)
        
        # Create DataFrames
        morgan_cols = [f"{smiles_col}_morgan_{i}" for i in range(n_comp_morgan)]
        maccs_cols = [f"{smiles_col}_maccs_{i}" for i in range(167)]
        rdkit_cols = [f"{smiles_col}_rdkit_{i}" for i in range(n_comp_rdkit)]
        desc_cols = [f"{smiles_col}_desc_{i}" for i in range(n_comp_desc)]
        
        morgan_df = pd.DataFrame(data=morgan_reduced, columns=morgan_cols)
        maccs_df = pd.DataFrame(data=maccs_fps, columns=maccs_cols)
        rdkit_df = pd.DataFrame(data=rdkit_reduced, columns=rdkit_cols)
        desc_df = pd.DataFrame(data=desc_reduced, columns=desc_cols)
        
        combined = pd.concat([morgan_df, maccs_df, rdkit_df, desc_df], axis=1)
        feature_list.append(combined)
    
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
    """Train and evaluate RandomForest with optimized parameters"""
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.15, random_state=random_state, shuffle=True
    )
    
    print(f"\n  Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"  Features: {X.shape[1]}")
    
    # Optimized RandomForest parameters
    print("\n  Training RandomForest...")
    rf_model = RandomForestRegressor(
        n_estimators=2000,          # Increased from 1000
        max_depth=30,               # Increased from 25
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        random_state=random_state,
        n_jobs=-1
    )
    
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=kf, scoring='r2')
    print(f"    CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"    CV R² scores: {cv_scores}")
    
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"    Test R²: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    
    plot_predictions(y_test, y_pred, target_name, r2, mae, rmse, "RandomForest_Optimized")
    
    return {
        "cv_r2": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "test_r2": r2,
        "mae": mae,
        "rmse": rmse
    }


def main():
    """Main function"""
    print("=" * 70)
    print("Optimized Modeling with RDKit Fingerprints")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    df = load_data()
    print(f"Dataset: {df.shape[0]} samples")
    
    # Generate enhanced features
    print("\nGenerating Enhanced Features...")
    X = generate_enhanced_features(df)
    print(f"\nTotal features: {X.shape[1]}")
    
    # Predict yield
    print("\n" + "=" * 70)
    print("Predicting YIELD")
    print("=" * 70)
    y_yield = df["yield"]
    yield_results = train_and_evaluate(X, y_yield, "Yield")
    
    # Predict ee
    print("\n" + "=" * 70)
    print("Predicting ENANTIOMERIC EXCESS (ee)")
    print("=" * 70)
    y_ee = df["ee"]
    ee_results = train_and_evaluate(X, y_ee, "Enantiomeric Excess")
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    print("\nYield Prediction Results:")
    marker = " ✓" if yield_results['cv_r2'] >= 0.6 else ""
    print(f"  RandomForest: CV R² = {yield_results['cv_r2']:.4f} (+/- {yield_results['cv_std']:.4f}), Test R² = {yield_results['test_r2']:.4f}{marker}")
    
    print("\nEnantiomeric Excess Prediction Results:")
    marker = " ✓" if ee_results['cv_r2'] >= 0.6 else ""
    print(f"  RandomForest: CV R² = {ee_results['cv_r2']:.4f} (+/- {ee_results['cv_std']:.4f}), Test R² = {ee_results['test_r2']:.4f}{marker}")
    
    print(f"\n{'='*70}")
    print(f"BEST CV R² RESULTS:")
    print(f"  Yield: {yield_results['cv_r2']:.4f}")
    print(f"  ee:    {ee_results['cv_r2']:.4f}")
    
    if yield_results['cv_r2'] >= 0.6 and ee_results['cv_r2'] >= 0.6:
        print("\n  ✅ BOTH TARGETS ACHIEVED CV R² >= 0.6!")
    elif yield_results['cv_r2'] >= 0.6:
        print("\n  ⚠️  Only Yield achieved CV R² >= 0.6")
    elif ee_results['cv_r2'] >= 0.6:
        print("\n  ⚠️  Only ee achieved CV R² >= 0.6")
    else:
        print(f"\n  ❌ Neither target achieved CV R² >= 0.6")
        print(f"     Yield gap: {0.6 - yield_results['cv_r2']:.4f}")
        print(f"     ee gap:    {0.6 - ee_results['cv_r2']:.4f}")
    
    print("=" * 70)


if __name__ == "__main__":
    main()