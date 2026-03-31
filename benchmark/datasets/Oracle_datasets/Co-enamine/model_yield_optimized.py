import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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
            morgan = smiles_to_fingerprints(smiles, radius=2, n_bits=2048)
            morgan_fps.append(morgan)
            
            maccs = smiles_to_maccs(smiles)
            maccs_fps.append(maccs)
            
            rdkit = smiles_to_rdkit_fp(smiles, n_bits=2048)
            rdkit_fps.append(rdkit)
            
            desc = smiles_to_descriptors(smiles)
            descriptors.append(desc)
        
        morgan_fps = np.array(morgan_fps)
        maccs_fps = np.array(maccs_fps)
        rdkit_fps = np.array(rdkit_fps)
        descriptors = np.array(descriptors)
        
        n_comp_morgan = min(100, morgan_fps.shape[1])
        n_comp_rdkit = min(100, rdkit_fps.shape[1])
        n_comp_desc = min(50, descriptors.shape[1])
        
        pca_morgan = PCA(n_components=n_comp_morgan, random_state=42)
        pca_rdkit = PCA(n_components=n_comp_rdkit, random_state=42)
        pca_desc = PCA(n_components=n_comp_desc, random_state=42)
        
        morgan_reduced = pca_morgan.fit_transform(morgan_fps)
        rdkit_reduced = pca_rdkit.fit_transform(rdkit_fps)
        desc_reduced = pca_desc.fit_transform(descriptors)
        
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


def train_and_evaluate_yield(X, y_yield, random_state=42):
    """Train and evaluate yield prediction with feature selection"""
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_yield, test_size=0.15, random_state=random_state, shuffle=True
    )
    
    print(f"\n  Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"  Features: {X_scaled.shape[1]}")
    
    results = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    
    # 1. RandomForest with feature selection
    print("\n  1. Training RandomForest with Feature Selection...")
    
    # First, do feature selection
    selector = SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=random_state), 
                               max_features=500, threshold=-np.inf)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    print(f"     Selected {X_train_selected.shape[1]} features")
    
    rf_params = {
        'n_estimators': [500, 1000, 1500],
        'max_depth': [15, 20, 25, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2', 0.3]
    }
    
    rf_base = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    rf_grid = GridSearchCV(rf_base, rf_params, cv=3, scoring='r2', n_jobs=-1, verbose=0)
    rf_grid.fit(X_train_selected, y_train)
    
    rf_best = rf_grid.best_estimator_
    cv_scores_rf = cross_val_score(rf_best, X_train_selected, y_train, cv=kf, scoring='r2')
    y_pred_rf = rf_best.predict(X_test_selected)
    
    results['RF_with_Selection'] = {
        'cv_r2': cv_scores_rf.mean(),
        'cv_std': cv_scores_rf.std(),
        'test_r2': r2_score(y_test, y_pred_rf),
        'best_params': rf_grid.best_params_
    }
    print(f"     Best params: {rf_grid.best_params_}")
    print(f"     CV R²: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std():.4f})")
    print(f"     Test R²: {r2_score(y_test, y_pred_rf):.4f}")
    
    # 2. Gradient Boosting
    print("\n  2. Training Gradient Boosting...")
    gb_params = {
        'n_estimators': [200, 500, 800],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.15],
        'min_samples_split': [2, 5],
        'subsample': [0.8, 1.0]
    }
    
    gb_base = GradientBoostingRegressor(random_state=random_state)
    gb_grid = GridSearchCV(gb_base, gb_params, cv=3, scoring='r2', n_jobs=-1, verbose=0)
    gb_grid.fit(X_train_selected, y_train)
    
    gb_best = gb_grid.best_estimator_
    cv_scores_gb = cross_val_score(gb_best, X_train_selected, y_train, cv=kf, scoring='r2')
    y_pred_gb = gb_best.predict(X_test_selected)
    
    results['GradientBoosting'] = {
        'cv_r2': cv_scores_gb.mean(),
        'cv_std': cv_scores_gb.std(),
        'test_r2': r2_score(y_test, y_pred_gb),
        'best_params': gb_grid.best_params_
    }
    print(f"     Best params: {gb_grid.best_params_}")
    print(f"     CV R²: {cv_scores_gb.mean():.4f} (+/- {cv_scores_gb.std():.4f})")
    print(f"     Test R²: {r2_score(y_test, y_pred_gb):.4f}")
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['cv_r2'])
    best_result = results[best_model_name]
    
    print(f"\n  Best Model for yield: {best_model_name}")
    print(f"  Best CV R²: {best_result['cv_r2']:.4f} (+/- {best_result['cv_std']:.4f})")
    
    return results, best_model_name


def main():
    """Main function"""
    print("=" * 70)
    print("Optimized Yield Modeling with Feature Selection")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    df = load_data()
    print(f"Dataset: {df.shape[0]} samples")
    
    # Generate enhanced features
    print("\nGenerating Enhanced Features...")
    X = generate_enhanced_features(df)
    print(f"\nTotal features: {X.shape[1]}")
    
    y_yield = df["yield"]
    
    # Predict yield with optimizations
    print("\n" + "=" * 70)
    print("Predicting YIELD with Optimizations")
    print("=" * 70)
    yield_results, yield_best = train_and_evaluate_yield(X, y_yield)
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    print("\nYield Prediction Results:")
    for name, res in yield_results.items():
        marker = " ✓" if res['cv_r2'] >= 0.6 else ""
        print(f"  {name:25s}: CV R² = {res['cv_r2']:.4f} (+/- {res['cv_std']:.4f}), Test R² = {res['test_r2']:.4f}{marker}")
    
    yield_cv_r2 = yield_results[yield_best]['cv_r2']
    
    print(f"\n{'='*70}")
    print(f"BEST Yield CV R²: {yield_cv_r2:.4f} (using {yield_best})")
    
    if yield_cv_r2 >= 0.6:
        print("\n  ✅ Yield ACHIEVED CV R² >= 0.6!")
    else:
        print(f"\n  ❌ Yield did not achieve CV R² >= 0.6")
        print(f"     Gap: {0.6 - yield_cv_r2:.4f}")
    
    print("=" * 70)


if __name__ == "__main__":
    main()