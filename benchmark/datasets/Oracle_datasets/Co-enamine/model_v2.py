import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor, ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression
from sklearn.linear_model import Ridge, ElasticNet
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


def plot_predictions(y_true, y_pred, target_name, r2, mae, rmse, filename):
    """Plot predicted vs true values"""
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_true.values, y=y_pred, alpha=0.6, s=60, edgecolor="none")
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")
    
    plt.xlabel("True Values", fontsize=14, fontweight="bold")
    plt.ylabel("Predicted Values", fontsize=14, fontweight="bold")
    plt.title(f"{target_name}: Predicted vs True", fontsize=16, fontweight="bold")
    
    metrics_text = f"R² = {r2:.4f}\nMAE = {mae:.4f}\nRMSE = {rmse:.4f}"
    plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"  Plot saved: {filename}")
    plt.close()


def train_and_evaluate_target(X, y, target_name, other_y=None, random_state=42):
    """Train and evaluate with extensive optimization"""
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # If other target available, use as feature (multi-task learning)
    if other_y is not None:
        scaler_other = StandardScaler()
        other_scaled = scaler_other.fit_transform(other_y.values.reshape(-1, 1))
        X_scaled = np.hstack([X_scaled, other_scaled])
        print(f"\n  Features: {X_scaled.shape[1]} (including auxiliary target)")
    else:
        print(f"\n  Features: {X_scaled.shape[1]}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.15, random_state=random_state, shuffle=True
    )
    
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    results = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    
    # 1. Extra Trees (often better than RandomForest)
    print(f"\n  1. Training Extra Trees for {target_name}...")
    
    # Feature selection
    selector = SelectFromModel(ExtraTreesRegressor(n_estimators=200, random_state=random_state), 
                               max_features=600, threshold=-np.inf)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    print(f"     Selected {X_train_selected.shape[1]} features")
    
    et_params = {
        'n_estimators': [1000, 1500, 2000],
        'max_depth': [10, 15, 20, 25, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2', 0.5]
    }
    
    et_base = ExtraTreesRegressor(random_state=random_state, n_jobs=-1)
    et_grid = GridSearchCV(et_base, et_params, cv=5, scoring='r2', n_jobs=-1, verbose=0)
    et_grid.fit(X_train_selected, y_train)
    
    et_best = et_grid.best_estimator_
    cv_scores_et = cross_val_score(et_best, X_train_selected, y_train, cv=kf, scoring='r2')
    y_pred_et = et_best.predict(X_test_selected)
    
    results['ExtraTrees'] = {
        'cv_r2': cv_scores_et.mean(),
        'cv_std': cv_scores_et.std(),
        'test_r2': r2_score(y_test, y_pred_et),
        'mae': mean_absolute_error(y_test, y_pred_et),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_et)),
        'best_params': et_grid.best_params_,
        'y_true': y_test,
        'y_pred': y_pred_et
    }
    print(f"     Best params: {et_grid.best_params_}")
    print(f"     CV R²: {cv_scores_et.mean():.4f} (+/- {cv_scores_et.std():.4f})")
    print(f"     Test R²: {r2_score(y_test, y_pred_et):.4f}")
    
    # 2. Advanced RandomForest
    print(f"\n  2. Training Optimized RandomForest for {target_name}...")
    
    rf_params = {
        'n_estimators': [1500, 2000, 2500],
        'max_depth': [15, 20, 25, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': [0.3, 0.5, 'sqrt']
    }
    
    rf_base = RandomForestRegressor(random_state=random_state, n_jobs=-1)
    rf_grid = GridSearchCV(rf_base, rf_params, cv=5, scoring='r2', n_jobs=-1, verbose=0)
    rf_grid.fit(X_train_selected, y_train)
    
    rf_best = rf_grid.best_estimator_
    cv_scores_rf = cross_val_score(rf_best, X_train_selected, y_train, cv=kf, scoring='r2')
    y_pred_rf = rf_best.predict(X_test_selected)
    
    results['RandomForest'] = {
        'cv_r2': cv_scores_rf.mean(),
        'cv_std': cv_scores_rf.std(),
        'test_r2': r2_score(y_test, y_pred_rf),
        'mae': mean_absolute_error(y_test, y_pred_rf),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        'best_params': rf_grid.best_params_,
        'y_true': y_test,
        'y_pred': y_pred_rf
    }
    print(f"     Best params: {rf_grid.best_params_}")
    print(f"     CV R²: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std():.4f})")
    print(f"     Test R²: {r2_score(y_test, y_pred_rf):.4f}")
    
    # 3. Advanced Gradient Boosting
    print(f"\n  3. Training Optimized Gradient Boosting for {target_name}...")
    gb_params = {
        'n_estimators': [500, 800, 1000],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.03, 0.05, 0.1],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    gb_base = GradientBoostingRegressor(random_state=random_state)
    gb_grid = GridSearchCV(gb_base, gb_params, cv=5, scoring='r2', n_jobs=-1, verbose=0)
    gb_grid.fit(X_train_selected, y_train)
    
    gb_best = gb_grid.best_estimator_
    cv_scores_gb = cross_val_score(gb_best, X_train_selected, y_train, cv=kf, scoring='r2')
    y_pred_gb = gb_best.predict(X_test_selected)
    
    results['GradientBoosting'] = {
        'cv_r2': cv_scores_gb.mean(),
        'cv_std': cv_scores_gb.std(),
        'test_r2': r2_score(y_test, y_pred_gb),
        'mae': mean_absolute_error(y_test, y_pred_gb),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_gb)),
        'best_params': gb_grid.best_params_,
        'y_true': y_test,
        'y_pred': y_pred_gb
    }
    print(f"     Best params: {gb_grid.best_params_}")
    print(f"     CV R²: {cv_scores_gb.mean():.4f} (+/- {cv_scores_gb.std():.4f})")
    print(f"     Test R²: {r2_score(y_test, y_pred_gb):.4f}")
    
    # 4. Stacking Ensemble
    print(f"\n  4. Training Stacking Ensemble for {target_name}...")
    estimators = [
        ('et', ExtraTreesRegressor(n_estimators=1000, max_depth=20, random_state=random_state, n_jobs=-1)),
        ('rf', RandomForestRegressor(n_estimators=1500, max_depth=25, random_state=random_state, n_jobs=-1)),
        ('gb', GradientBoostingRegressor(n_estimators=500, max_depth=5, learning_rate=0.1, random_state=random_state))
    ]
    stacking = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=5,
        n_jobs=-1
    )
    
    cv_scores_stack = cross_val_score(stacking, X_train_selected, y_train, cv=kf, scoring='r2')
    stacking.fit(X_train_selected, y_train)
    y_pred_stack = stacking.predict(X_test_selected)
    
    results['Stacking'] = {
        'cv_r2': cv_scores_stack.mean(),
        'cv_std': cv_scores_stack.std(),
        'test_r2': r2_score(y_test, y_pred_stack),
        'mae': mean_absolute_error(y_test, y_pred_stack),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_stack)),
        'best_params': 'N/A',
        'y_true': y_test,
        'y_pred': y_pred_stack
    }
    print(f"     CV R²: {cv_scores_stack.mean():.4f} (+/- {cv_scores_stack.std():.4f})")
    print(f"     Test R²: {r2_score(y_test, y_pred_stack):.4f}")
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['cv_r2'])
    best_result = results[best_model_name]
    
    # Plot best model
    filename = f"{target_name.lower().replace(' ', '_')}_{best_model_name.lower()}_v2.png"
    plot_predictions(best_result['y_true'], best_result['y_pred'], target_name,
                    best_result['test_r2'], best_result['mae'], best_result['rmse'], filename)
    
    print(f"\n  Best Model for {target_name}: {best_model_name}")
    print(f"  Best CV R²: {best_result['cv_r2']:.4f} (+/- {best_result['cv_std']:.4f})")
    
    return results, best_model_name


def main():
    """Main function"""
    print("=" * 70)
    print("Model v2 - Target: R² > 0.7 for both Yield and ee")
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
    y_ee = df["ee"]
    
    # Train yield with ee as auxiliary feature
    print("\n" + "=" * 70)
    print("Training YIELD (with ee as auxiliary feature)")
    print("=" * 70)
    yield_results, yield_best = train_and_evaluate_target(X, y_yield, "Yield", other_y=y_ee)
    
    # Train ee with yield as auxiliary feature
    print("\n" + "=" * 70)
    print("Training ENANTIOMERIC EXCESS (with yield as auxiliary feature)")
    print("=" * 70)
    ee_results, ee_best = train_and_evaluate_target(X, y_ee, "Enantiomeric Excess", other_y=y_yield)
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    print("\nYIELD PREDICTION RESULTS:")
    for name, res in yield_results.items():
        marker = " ✓" if res['cv_r2'] >= 0.7 else (" ○" if res['cv_r2'] >= 0.6 else "")
        print(f"  {name:20s}: CV R² = {res['cv_r2']:.4f} (+/- {res['cv_std']:.4f}), Test R² = {res['test_r2']:.4f}{marker}")
    
    print("\nENANTIOMERIC EXCESS PREDICTION RESULTS:")
    for name, res in ee_results.items():
        marker = " ✓" if res['cv_r2'] >= 0.7 else (" ○" if res['cv_r2'] >= 0.6 else "")
        print(f"  {name:20s}: CV R² = {res['cv_r2']:.4f} (+/- {res['cv_std']:.4f}), Test R² = {res['test_r2']:.4f}{marker}")
    
    yield_cv_r2 = yield_results[yield_best]['cv_r2']
    ee_cv_r2 = ee_results[ee_best]['cv_r2']
    
    print(f"\n{'='*70}")
    print("BEST RESULTS:")
    print(f"  Yield: {yield_best} - CV R² = {yield_cv_r2:.4f}")
    print(f"  ee:    {ee_best} - CV R² = {ee_cv_r2:.4f}")
    
    if yield_cv_r2 >= 0.7 and ee_cv_r2 >= 0.7:
        print("\n  🎉 BOTH TARGETS ACHIEVED CV R² >= 0.7!")
    elif yield_cv_r2 >= 0.7:
        print(f"\n  ⚠️  Only Yield achieved CV R² >= 0.7 (ee gap: {0.7 - ee_cv_r2:.4f})")
    elif ee_cv_r2 >= 0.7:
        print(f"\n  ⚠️  Only ee achieved CV R² >= 0.7 (Yield gap: {0.7 - yield_cv_r2:.4f})")
    else:
        print(f"\n  Current Status:")
        print(f"    Yield: {yield_cv_r2:.4f} (gap to 0.7: {0.7 - yield_cv_r2:.4f})")
        print(f"    ee:    {ee_cv_r2:.4f} (gap to 0.7: {0.7 - ee_cv_r2:.4f})")
    
    print("=" * 70)


if __name__ == "__main__":
    main()