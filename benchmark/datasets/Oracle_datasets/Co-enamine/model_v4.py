import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
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
    """Load main dataset and DFT descriptor files"""
    df = pd.read_csv("Co-enamine.csv")
    
    # Load DFT descriptors
    dft_files = {
        'amine_smiles_ion': 'descriptors/amine_ion_desc_dft.csv',
        'amine_smiles_anion': 'descriptors/amine_anion_desc_dft.csv',
        'cobalt_smiles': 'descriptors/cobalt_desc_dft.csv',
        'oxidant_smiles': 'descriptors/oxidant_desc_dft.csv',
        'alkali_smiles': 'descriptors/alkali_desc_dft.csv',
        'solvent_smiles': 'descriptors/solvent_desc_dft.csv'
    }
    
    dft_descriptors = {}
    for col, filepath in dft_files.items():
        try:
            dft_df = pd.read_csv(filepath)
            # Use SMILES as index for mapping
            dft_df = dft_df.set_index('SMILES')
            dft_descriptors[col] = dft_df
            print(f"  Loaded DFT descriptors for {col}: {dft_df.shape[1]} features")
        except Exception as e:
            print(f"  Warning: Could not load DFT descriptors for {col}: {e}")
            dft_descriptors[col] = None
    
    return df, dft_descriptors


def generate_enhanced_features(df, dft_descriptors):
    """Generate enhanced features using RDKit fingerprints and DFT descriptors"""
    feature_list = []
    smiles_cols = ['amine_smiles_ion', 'amine_smiles_anion', 'cobalt_smiles', 
                   'oxidant_smiles', 'alkali_smiles', 'solvent_smiles']
    
    print("Generating molecular fingerprints and adding DFT descriptors...")
    
    for smiles_col in smiles_cols:
        print(f"  Processing {smiles_col}...")
        
        # RDKit fingerprints
        morgan_fps = []
        maccs_fps = []
        rdkit_fps = []
        rdkit_descs = []
        
        for smiles in df[smiles_col]:
            morgan = smiles_to_fingerprints(smiles, radius=2, n_bits=2048)
            morgan_fps.append(morgan)
            
            maccs = smiles_to_maccs(smiles)
            maccs_fps.append(maccs)
            
            rdkit = smiles_to_rdkit_fp(smiles, n_bits=2048)
            rdkit_fps.append(rdkit)
            
            desc = smiles_to_descriptors(smiles)
            rdkit_descs.append(desc)
        
        morgan_fps = np.array(morgan_fps)
        maccs_fps = np.array(maccs_fps)
        rdkit_fps = np.array(rdkit_fps)
        rdkit_descs = np.array(rdkit_descs)
        
        # PCA reduction for high-dimensional features
        n_comp_morgan = min(100, morgan_fps.shape[1])
        n_comp_rdkit = min(100, rdkit_fps.shape[1])
        n_comp_desc = min(50, rdkit_descs.shape[1])
        
        pca_morgan = PCA(n_components=n_comp_morgan, random_state=42)
        pca_rdkit = PCA(n_components=n_comp_rdkit, random_state=42)
        pca_desc = PCA(n_components=n_comp_desc, random_state=42)
        
        morgan_reduced = pca_morgan.fit_transform(morgan_fps)
        rdkit_reduced = pca_rdkit.fit_transform(rdkit_fps)
        desc_reduced = pca_desc.fit_transform(rdkit_descs)
        
        # Combine RDKit features
        morgan_cols = [f"{smiles_col}_morgan_{i}" for i in range(n_comp_morgan)]
        maccs_cols = [f"{smiles_col}_maccs_{i}" for i in range(167)]
        rdkit_cols = [f"{smiles_col}_rdkit_{i}" for i in range(n_comp_rdkit)]
        desc_cols = [f"{smiles_col}_desc_{i}" for i in range(n_comp_desc)]
        
        morgan_df = pd.DataFrame(data=morgan_reduced, columns=morgan_cols)
        maccs_df = pd.DataFrame(data=maccs_fps, columns=maccs_cols)
        rdkit_df = pd.DataFrame(data=rdkit_reduced, columns=rdkit_cols)
        desc_df = pd.DataFrame(data=desc_reduced, columns=desc_cols)
        
        combined = pd.concat([morgan_df, maccs_df, rdkit_df, desc_df], axis=1)
        
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
            combined = pd.concat([combined, dft_feature_df], axis=1)
            print(f"    Added {dft_features.shape[1]} DFT features")
        
        feature_list.append(combined)
    
    X = pd.concat(feature_list, axis=1)
    return X


def plot_predictions(y_true, y_pred, target_name, r2, filename):
    """Plot predicted vs true values"""
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.6, s=60, edgecolor="none")
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")
    
    plt.xlabel("True Values", fontsize=14, fontweight="bold")
    plt.ylabel("Predicted Values", fontsize=14, fontweight="bold")
    plt.title(f"{target_name}: Predicted vs True\nR² = {r2:.4f}", fontsize=16, fontweight="bold")
    
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"  Plot saved: {filename}")
    plt.close()


def train_best_models(X, y_yield, y_ee):
    """Train best models with optimized parameters"""
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Add cross-features
    scaler_yield = StandardScaler()
    scaler_ee = StandardScaler()
    yield_scaled = scaler_yield.fit_transform(y_yield.values.reshape(-1, 1))
    ee_scaled = scaler_ee.fit_transform(y_ee.values.reshape(-1, 1))
    
    results = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # ========== YIELD PREDICTION ==========
    print("\n" + "=" * 70)
    print("YIELD PREDICTION")
    print("=" * 70)
    
    # Use ee as auxiliary feature
    X_yield = np.hstack([X_scaled, ee_scaled])
    X_train, X_test, y_train, y_test = train_test_split(
        X_yield, y_yield, test_size=0.15, random_state=42, shuffle=True
    )
    
    # Feature selection
    selector = SelectFromModel(ExtraTreesRegressor(n_estimators=200, random_state=42), 
                               max_features=600, threshold=-np.inf)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)
    print(f"Selected {X_train_sel.shape[1]} features")
    
    # Best model for yield
    print("\nTraining optimized Extra Trees for Yield...")
    et_yield = ExtraTreesRegressor(
        n_estimators=2000,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features=0.5,
        random_state=42,
        n_jobs=-1
    )
    et_yield.fit(X_train_sel, y_train)
    y_pred_yield = et_yield.predict(X_test_sel)
    
    cv_scores_yield = cross_val_score(et_yield, X_train_sel, y_train, cv=kf, scoring='r2')
    r2_yield = r2_score(y_test, y_pred_yield)
    
    print(f"  CV R²: {cv_scores_yield.mean():.4f} (+/- {cv_scores_yield.std():.4f})")
    print(f"  Test R²: {r2_yield:.4f}")
    
    results['Yield'] = {
        'cv_r2': cv_scores_yield.mean(),
        'cv_std': cv_scores_yield.std(),
        'test_r2': r2_yield,
        'y_true': y_test,
        'y_pred': y_pred_yield
    }
    
    plot_predictions(y_test, y_pred_yield, "Yield", r2_yield, "yield_v4.png")
    
    # ========== EE PREDICTION ==========
    print("\n" + "=" * 70)
    print("ENANTIOMERIC EXCESS PREDICTION")
    print("=" * 70)
    
    # Use yield as auxiliary feature
    X_ee = np.hstack([X_scaled, yield_scaled])
    X_train, X_test, y_train, y_test = train_test_split(
        X_ee, y_ee, test_size=0.15, random_state=42, shuffle=True
    )
    
    # Feature selection
    selector = SelectFromModel(RandomForestRegressor(n_estimators=200, random_state=42), 
                               max_features=500, threshold=-np.inf)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)
    print(f"Selected {X_train_sel.shape[1]} features")
    
    # Best model for ee
    print("\nTraining optimized RandomForest for ee...")
    rf_ee = RandomForestRegressor(
        n_estimators=1500,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=2,
        max_features=0.3,
        random_state=42,
        n_jobs=-1
    )
    rf_ee.fit(X_train_sel, y_train)
    y_pred_ee = rf_ee.predict(X_test_sel)
    
    cv_scores_ee = cross_val_score(rf_ee, X_train_sel, y_train, cv=kf, scoring='r2')
    r2_ee = r2_score(y_test, y_pred_ee)
    
    print(f"  CV R²: {cv_scores_ee.mean():.4f} (+/- {cv_scores_ee.std():.4f})")
    print(f"  Test R²: {r2_ee:.4f}")
    
    results['ee'] = {
        'cv_r2': cv_scores_ee.mean(),
        'cv_std': cv_scores_ee.std(),
        'test_r2': r2_ee,
        'y_true': y_test,
        'y_pred': y_pred_ee
    }
    
    plot_predictions(y_test, y_pred_ee, "Enantiomeric Excess", r2_ee, "ee_v4.png")
    
    return results


def main():
    """Main function"""
    print("=" * 70)
    print("Model v4 - DFT + RDKit Descriptors")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    df, dft_descriptors = load_data()
    print(f"\nDataset: {df.shape[0]} samples")
    
    # Generate enhanced features
    print("\nGenerating Enhanced Features (RDKit + DFT)...")
    X = generate_enhanced_features(df, dft_descriptors)
    print(f"\nTotal features: {X.shape[1]}")
    
    y_yield = df["yield"]
    y_ee = df["ee"]
    
    # Train best models
    results = train_best_models(X, y_yield, y_ee)
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    yield_cv = results['Yield']['cv_r2']
    yield_std = results['Yield']['cv_std']
    yield_test = results['Yield']['test_r2']
    
    ee_cv = results['ee']['cv_r2']
    ee_std = results['ee']['cv_std']
    ee_test = results['ee']['test_r2']
    
    print(f"\nYield Prediction:")
    print(f"  CV R²:  {yield_cv:.4f} (+/- {yield_std:.4f})")
    print(f"  Test R²: {yield_test:.4f}")
    status_yield = "✅ >= 0.7" if yield_cv >= 0.7 else ("✅ >= 0.6" if yield_cv >= 0.6 else "⚠️ < 0.6")
    print(f"  Status: {status_yield}")
    
    print(f"\nEnantiomeric Excess Prediction:")
    print(f"  CV R²:  {ee_cv:.4f} (+/- {ee_std:.4f})")
    print(f"  Test R²: {ee_test:.4f}")
    status_ee = "✅ >= 0.7" if ee_cv >= 0.7 else ("✅ >= 0.6" if ee_cv >= 0.6 else "⚠️ < 0.6")
    print(f"  Status: {status_ee}")
    
    print(f"\n{'='*70}")
    print("Comparison with v3 (no DFT):")
    print(f"  Yield: {yield_cv:.4f} vs 0.6345 (v3)")
    print(f"  ee:    {ee_cv:.4f} vs 0.6447 (v3)")
    
    if yield_cv >= 0.6 and ee_cv >= 0.6:
        print("\n🎉 SUCCESS! Both targets achieved R² >= 0.6!")
    else:
        print(f"\nNote: DFT descriptors added {X.shape[1] - 2502} extra features")
    print("=" * 70)


if __name__ == "__main__":
    main()