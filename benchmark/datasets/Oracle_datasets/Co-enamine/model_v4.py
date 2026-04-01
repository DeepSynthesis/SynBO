import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.neural_network import MLPRegressor
from rdkit import Chem
from rdkit.Chem import Descriptors

# Try to import XGBoost and CatBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not available")

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")


def smiles_to_descriptors(smiles):
    """Generate RDKit molecular descriptors only"""
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


def generate_features(df, dft_descriptors):
    """Generate features using RDKit descriptors and DFT descriptors only"""
    feature_list = []
    smiles_cols = ['amine_smiles_ion', 'amine_smiles_anion', 'cobalt_smiles', 
                   'oxidant_smiles', 'alkali_smiles', 'solvent_smiles']
    
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


def evaluate_model(model, X, y, model_name, target_name, kf):
    """Evaluate model using 5-fold CV and return results"""
    print(f"\n  Training {model_name}...")
    
    # 5-fold CV scores
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='r2', n_jobs=-1)
    cv_r2_mean = cv_scores.mean()
    cv_r2_std = cv_scores.std()
    
    # Cross-validated predictions for plotting
    y_pred = cross_val_predict(model, X, y, cv=kf, n_jobs=-1)
    overall_r2 = r2_score(y, y_pred)
    
    print(f"    5-Fold CV R²: {cv_r2_mean:.4f} (+/- {cv_r2_std:.4f})")
    print(f"    Overall R² (all data): {overall_r2:.4f}")
    
    return {
        'cv_r2_mean': cv_r2_mean,
        'cv_r2_std': cv_r2_std,
        'overall_r2': overall_r2,
        'y_true': y,
        'y_pred': y_pred,
        'model': model
    }


def train_models_5fold_cv(X, y_yield, y_ee):
    """Train multiple models using 5-fold CV on full dataset"""
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Add cross-features (ee for yield prediction, yield for ee prediction)
    scaler_yield = StandardScaler()
    scaler_ee = StandardScaler()
    yield_scaled = scaler_yield.fit_transform(y_yield.values.reshape(-1, 1))
    ee_scaled = scaler_ee.fit_transform(y_ee.values.reshape(-1, 1))
    
    results = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # ========== YIELD PREDICTION ==========
    print("\n" + "=" * 70)
    print("YIELD PREDICTION - 5-Fold CV")
    print("=" * 70)
    
    # Use ee as auxiliary feature
    X_yield = np.hstack([X_scaled, ee_scaled])
    
    # Feature selection
    print("\n  Feature selection...")
    selector = SelectFromModel(ExtraTreesRegressor(n_estimators=200, random_state=42), 
                               max_features=600, threshold=-np.inf)
    X_yield_sel = selector.fit_transform(X_yield, y_yield)
    print(f"  Selected {X_yield_sel.shape[1]} features")
    
    # Define models for yield
    yield_models = {
        'ExtraTrees': ExtraTreesRegressor(
            n_estimators=2000, max_depth=20, min_samples_split=5, 
            min_samples_leaf=2, max_features=0.5, random_state=42, n_jobs=-1
        ),
        'RandomForest': RandomForestRegressor(
            n_estimators=1500, max_depth=20, min_samples_split=5, 
            min_samples_leaf=2, max_features=0.3, random_state=42, n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05, 
            min_samples_split=5, random_state=42
        )
    }
    
    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        yield_models['XGBoost'] = xgb.XGBRegressor(
            n_estimators=1000, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
        )
    
    # Add CatBoost if available
    if CATBOOST_AVAILABLE:
        yield_models['CatBoost'] = cb.CatBoostRegressor(
            iterations=1000, depth=8, learning_rate=0.05,
            verbose=False, random_state=42
        )
    
    # Add Neural Network
    yield_models['NeuralNetwork'] = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64), max_iter=1000,
        early_stopping=True, validation_fraction=0.1,
        random_state=42
    )
    
    # Evaluate all models
    yield_results = {}
    best_yield_r2 = -np.inf
    best_yield_model_name = None
    
    for name, model in yield_models.items():
        result = evaluate_model(model, X_yield_sel, y_yield, name, "Yield", kf)
        yield_results[name] = result
        
        if result['cv_r2_mean'] > best_yield_r2:
            best_yield_r2 = result['cv_r2_mean']
            best_yield_model_name = name
    
    # Plot best model
    best_result = yield_results[best_yield_model_name]
    plot_cv_predictions(
        best_result['y_true'], best_result['y_pred'], 
        f"Yield ({best_yield_model_name})", best_result['overall_r2'], 
        "yield_v4.png"
    )
    
    results['Yield'] = {
        'best_model': best_yield_model_name,
        'best_cv_r2': best_yield_r2,
        'all_results': yield_results
    }
    
    # ========== EE PREDICTION ==========
    print("\n" + "=" * 70)
    print("ENANTIOMERIC EXCESS PREDICTION - 5-Fold CV")
    print("=" * 70)
    
    # Use yield as auxiliary feature
    X_ee = np.hstack([X_scaled, yield_scaled])
    
    # Feature selection
    print("\n  Feature selection...")
    selector = SelectFromModel(RandomForestRegressor(n_estimators=200, random_state=42), 
                               max_features=500, threshold=-np.inf)
    X_ee_sel = selector.fit_transform(X_ee, y_ee)
    print(f"  Selected {X_ee_sel.shape[1]} features")
    
    # Define models for ee
    ee_models = {
        'RandomForest': RandomForestRegressor(
            n_estimators=1500, max_depth=20, min_samples_split=10, 
            min_samples_leaf=2, max_features=0.3, random_state=42, n_jobs=-1
        ),
        'ExtraTrees': ExtraTreesRegressor(
            n_estimators=2000, max_depth=20, min_samples_split=5, 
            min_samples_leaf=2, max_features=0.5, random_state=42, n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05, 
            min_samples_split=5, random_state=42
        )
    }
    
    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        ee_models['XGBoost'] = xgb.XGBRegressor(
            n_estimators=1000, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
        )
    
    # Add CatBoost if available
    if CATBOOST_AVAILABLE:
        ee_models['CatBoost'] = cb.CatBoostRegressor(
            iterations=1000, depth=8, learning_rate=0.05,
            verbose=False, random_state=42
        )
    
    # Add Neural Network
    ee_models['NeuralNetwork'] = MLPRegressor(
        hidden_layer_sizes=(256, 128, 64), max_iter=1000,
        early_stopping=True, validation_fraction=0.1,
        random_state=42
    )
    
    # Evaluate all models
    ee_results = {}
    best_ee_r2 = -np.inf
    best_ee_model_name = None
    
    for name, model in ee_models.items():
        result = evaluate_model(model, X_ee_sel, y_ee, name, "ee", kf)
        ee_results[name] = result
        
        if result['cv_r2_mean'] > best_ee_r2:
            best_ee_r2 = result['cv_r2_mean']
            best_ee_model_name = name
    
    # Plot best model
    best_result = ee_results[best_ee_model_name]
    plot_cv_predictions(
        best_result['y_true'], best_result['y_pred'], 
        f"Enantiomeric Excess ({best_ee_model_name})", best_result['overall_r2'], 
        "ee_v4.png"
    )
    
    results['ee'] = {
        'best_model': best_ee_model_name,
        'best_cv_r2': best_ee_r2,
        'all_results': ee_results
    }
    
    return results


def main():
    """Main function"""
    print("=" * 70)
    print("Model v4 - RDKit Descriptors + DFT Descriptors (5-Fold CV)")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    df, dft_descriptors = load_data()
    print(f"\nDataset: {df.shape[0]} samples")
    
    # Generate features
    print("\nGenerating Features (RDKit Descriptors + DFT)...")
    X = generate_features(df, dft_descriptors)
    print(f"\nTotal features: {X.shape[1]}")
    
    y_yield = df["yield"]
    y_ee = df["ee"]
    
    # Train models with 5-fold CV
    results = train_models_5fold_cv(X, y_yield, y_ee)
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY - 5-Fold CV Results")
    print("=" * 70)
    
    # Yield results
    print("\nYield Prediction Results:")
    yield_best = results['Yield']['best_model']
    yield_best_r2 = results['Yield']['best_cv_r2']
    print(f"  Best Model: {yield_best}")
    print(f"  Best CV R²: {yield_best_r2:.4f}")
    
    print("\n  All Models:")
    for name, result in results['Yield']['all_results'].items():
        status = "✅" if result['cv_r2_mean'] >= 0.6 else "⚠️"
        print(f"    {status} {name:20s}: CV R² = {result['cv_r2_mean']:.4f} (+/- {result['cv_r2_std']:.4f})")
    
    # ee results
    print("\nEnantiomeric Excess Prediction Results:")
    ee_best = results['ee']['best_model']
    ee_best_r2 = results['ee']['best_cv_r2']
    print(f"  Best Model: {ee_best}")
    print(f"  Best CV R²: {ee_best_r2:.4f}")
    
    print("\n  All Models:")
    for name, result in results['ee']['all_results'].items():
        status = "✅" if result['cv_r2_mean'] >= 0.6 else "⚠️"
        print(f"    {status} {name:20s}: CV R² = {result['cv_r2_mean']:.4f} (+/- {result['cv_r2_std']:.4f})")
    
    # Overall status
    print(f"\n{'='*70}")
    if yield_best_r2 >= 0.6 and ee_best_r2 >= 0.6:
        print("🎉 SUCCESS! Both targets achieved 5-Fold CV R² >= 0.6!")
    else:
        print("⚠️  Some targets did not reach R² >= 0.6")
        if yield_best_r2 < 0.6:
            print(f"   Yield gap: {0.6 - yield_best_r2:.4f}")
        if ee_best_r2 < 0.6:
            print(f"   ee gap: {0.6 - ee_best_r2:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()