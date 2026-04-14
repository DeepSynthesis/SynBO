"""Suzuki HTE - XGBoost 5折交叉验证"""
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BEST_PARAMS = {
    'n_estimators': 1274, 'max_depth': 8,
    'learning_rate': 0.018029253904222754,
    'subsample': 0.8530597199777481,
    'colsample_bytree': 0.9505932960648681,
    'min_child_weight': 3, 'reg_alpha': 5.091542017959273,
    'reg_lambda': 4.5283455254313336e-08, 'gamma': 1.1504135041788988,
}

def get_rdkit_desc_names():
    return [desc_name for desc_name, _ in Descriptors._descList]

def compute_rdkit_descriptors(smiles, calculator, n_desc):
    if pd.isna(smiles) or str(smiles).strip().lower() in ["blank_cell", "blank", "na", "nan", ""]:
        return np.zeros(n_desc)
    mol = Chem.MolFromSmiles(str(smiles).split("~")[0])
    if mol is None:
        return np.zeros(n_desc)
    try:
        return np.array(calculator.CalcDescriptors(mol))
    except:
        return np.zeros(n_desc)

def process_column_features(series, desc_names, calculator):
    unique_smiles = series.dropna().unique()
    desc_dict = {sm: compute_rdkit_descriptors(sm, calculator, len(desc_names)) for sm in unique_smiles}
    return np.array([desc_dict.get(sm, np.zeros(len(desc_names))) if pd.notna(sm) else np.zeros(len(desc_names)) for sm in series])

def load_and_prepare_data():
    print("=" * 70)
    print("Suzuki HTE - XGBoost 5折交叉验证")
    print("=" * 70)
    df = pd.read_csv("suzuki_HTE.csv")
    print(f"数据集: {df.shape[0]} 条记录")
    y = df["Conversion"].values
    desc_names = get_rdkit_desc_names()
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
    mol_types = ["solvent", "ligand", "reactant2", "reactant1", "base"]
    print(f"提取RDKit描述符: {mol_types}")
    all_features = [process_column_features(df[mt], desc_names, calculator) for mt in mol_types]
    X = np.hstack(all_features)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    valid_features = np.var(X, axis=0) > 1e-10
    X = X[:, valid_features]
    print(f"特征矩阵形状: {X.shape}")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0), y, scaler, df

def perform_cv(X, y):
    print(f"\n最佳参数: {BEST_PARAMS}")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {'train_r2': [], 'val_r2': [], 'val_rmse': [], 'val_mae': [], 'y_test_all': [], 'y_pred_all': []}
    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model = xgb.XGBRegressor(**BEST_PARAMS, random_state=42, n_jobs=-1, verbosity=0, tree_method='hist')
        model.fit(X_train, y_train)
        y_train_pred, y_test_pred = model.predict(X_train), model.predict(X_test)
        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_test, y_test_pred)
        val_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        val_mae = mean_absolute_error(y_test, y_test_pred)
        print(f"Fold {fold_idx+1}: Train R²={train_r2:.4f}, Val R²={val_r2:.4f}, RMSE={val_rmse:.4f}")
        results['train_r2'].append(train_r2)
        results['val_r2'].append(val_r2)
        results['val_rmse'].append(val_rmse)
        results['val_mae'].append(val_mae)
        results['y_test_all'].extend(y_test.tolist())
        results['y_pred_all'].extend(y_test_pred.tolist())
    return results

def plot_predictions(y_true, y_pred, save_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='none', s=50, c='steelblue')
    min_v, max_v = min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))
    margin = (max_v - min_v) * 0.05
    ax.plot([min_v-margin, max_v+margin], [min_v-margin, max_v+margin], 'r--', lw=2, label='Perfect Prediction')
    r2 = r2_score(y_true, y_pred)
    ax.set_xlabel('Experimental Conversion (%)', fontsize=12)
    ax.set_ylabel('Predicted Conversion (%)', fontsize=12)
    ax.set_title(f'XGBoost 5-Fold CV Results\nCV R² = {r2:.4f}', fontsize=14)
    ax.legend(loc='upper left')
    ax.set_xlim(min_v-margin, max_v+margin)
    ax.set_ylim(min_v-margin, max_v+margin)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    rmse, mae = np.sqrt(mean_squared_error(y_true, y_pred)), mean_absolute_error(y_true, y_pred)
    ax.text(0.95, 0.05, f'RMSE: {rmse:.3f}\nMAE: {mae:.3f}\nn={len(y_true)}', transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图片已保存: {save_path}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    X, y, scaler, df = load_and_prepare_data()
    results = perform_cv(X, y)
    cv_r2 = np.mean(results['val_r2'])
    print(f"\n{'='*50}\nCV R² (Mean±Std): {cv_r2:.4f} ± {np.std(results['val_r2']):.4f}")
    print(f"CV RMSE (Mean±Std): {np.mean(results['val_rmse']):.4f} ± {np.std(results['val_rmse']):.4f}")
    print(f"CV MAE (Mean±Std): {np.mean(results['val_mae']):.4f} ± {np.std(results['val_mae']):.4f}")
    plot_predictions(np.array(results['y_test_all']), np.array(results['y_pred_all']), os.path.join(script_dir, "cv_prediction_results.png"))
    with open(os.path.join(script_dir, "cv_results_summary.txt"), 'w') as f:
        f.write(f"CV R²: {cv_r2:.4f}\nBest params: {BEST_PARAMS}\n")
    print("✅ 5折交叉验证完成!")
    return cv_r2

if __name__ == "__main__":
    main()
