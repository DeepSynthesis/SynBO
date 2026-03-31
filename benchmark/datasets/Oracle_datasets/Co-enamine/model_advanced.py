import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 导入 XGBoost 和 CatBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available")

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available")

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
        else:
            cols = [f"{smiles_col}_feat{i}" for i in range(matched_desc.shape[1])]
            desc_df = pd.DataFrame(matched_desc, columns=cols)
            feature_list.append(desc_df)
    
    X = pd.concat(feature_list, axis=1)
    return X


class DeepNN(nn.Module):
    """Deep Neural Network for regression"""
    def __init__(self, input_size, hidden_sizes=[256, 128, 64, 32]):
        super(DeepNN, self).__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.BatchNorm1d(hidden_size)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def train_pytorch_model(X_train, y_train, X_val, y_val, input_size, epochs=500, patience=50):
    """Train PyTorch model with early stopping"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train.values).view(-1, 1).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val.values).view(-1, 1).to(device)
    
    # Create model
    model = DeepNN(input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    
    best_val_loss = float('inf')
    best_model_state = model.state_dict()
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model


def evaluate_pytorch_model(model, X, y, device):
    """Evaluate PyTorch model"""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        predictions = model(X_tensor).cpu().numpy().flatten()
    return predictions


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


def train_and_evaluate_models(X, y, target_name, random_state=42):
    """Train and evaluate multiple advanced models"""
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.15, random_state=random_state, shuffle=True
    )
    
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    results = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. XGBoost with optimized parameters
    if XGBOOST_AVAILABLE:
        print("\n  Training XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=1000,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=random_state,
            n_jobs=-1
        )
        
        kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=kf, scoring='r2')
        print(f"    CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"    Test R²: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        
        results["XGBoost"] = {
            "model": xgb_model,
            "cv_r2": cv_scores.mean(),
            "test_r2": r2,
            "mae": mae,
            "rmse": rmse,
            "y_pred": y_pred,
            "y_test": y_test
        }
        plot_predictions(y_test, y_pred, target_name, r2, mae, rmse, "XGBoost")
    
    # 2. CatBoost with optimized parameters
    if CATBOOST_AVAILABLE:
        print("\n  Training CatBoost...")
        cat_model = CatBoostRegressor(
            iterations=1000,
            depth=8,
            learning_rate=0.05,
            l2_leaf_reg=3,
            random_seed=random_state,
            verbose=False,
            loss_function='RMSE'
        )
        
        kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(cat_model, X_train, y_train, cv=kf, scoring='r2')
        print(f"    CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        cat_model.fit(X_train, y_train)
        y_pred = cat_model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"    Test R²: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        
        results["CatBoost"] = {
            "model": cat_model,
            "cv_r2": cv_scores.mean(),
            "test_r2": r2,
            "mae": mae,
            "rmse": rmse,
            "y_pred": y_pred,
            "y_test": y_test
        }
        plot_predictions(y_test, y_pred, target_name, r2, mae, rmse, "CatBoost")
    
    # 3. RandomForest (tuned)
    print("\n  Training RandomForest...")
    rf_model = RandomForestRegressor(
        n_estimators=1000,
        max_depth=20,
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
    
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"    Test R²: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    
    results["RandomForest"] = {
        "model": rf_model,
        "cv_r2": cv_scores.mean(),
        "test_r2": r2,
        "mae": mae,
        "rmse": rmse,
        "y_pred": y_pred,
        "y_test": y_test
    }
    plot_predictions(y_test, y_pred, target_name, r2, mae, rmse, "RandomForest")
    
    # 4. PyTorch Deep Neural Network
    print("\n  Training PyTorch Deep NN...")
    
    # Manual 5-fold CV for PyTorch
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores_pytorch = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = train_pytorch_model(X_fold_train, y_fold_train, X_fold_val, y_fold_val, 
                                    X_train.shape[1], epochs=500, patience=50)
        y_fold_pred = evaluate_pytorch_model(model, X_fold_val, y_fold_val, device)
        fold_r2 = r2_score(y_fold_val, y_fold_pred)
        cv_scores_pytorch.append(fold_r2)
    
    cv_scores_pytorch = np.array(cv_scores_pytorch)
    print(f"    CV R²: {cv_scores_pytorch.mean():.4f} (+/- {cv_scores_pytorch.std():.4f})")
    
    # Train final model on all training data
    final_model = train_pytorch_model(X_train, y_train, X_test, y_test, 
                                      X_train.shape[1], epochs=1000, patience=100)
    y_pred = evaluate_pytorch_model(final_model, X_test, y_test, device)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"    Test R²: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    
    results["PyTorch_DNN"] = {
        "model": final_model,
        "cv_r2": cv_scores_pytorch.mean(),
        "test_r2": r2,
        "mae": mae,
        "rmse": rmse,
        "y_pred": y_pred,
        "y_test": y_test
    }
    plot_predictions(y_test, y_pred, target_name, r2, mae, rmse, "PyTorch_DNN")
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['cv_r2'])
    print(f"\n  Best Model: {best_model_name}")
    print(f"  Best CV R²: {results[best_model_name]['cv_r2']:.4f}")
    print(f"  Best Test R²: {results[best_model_name]['test_r2']:.4f}")
    
    return results, best_model_name


def main():
    """Main function"""
    print("=" * 70)
    print("Advanced Modeling with XGBoost, CatBoost, and PyTorch")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    df, descriptors = load_data()
    print(f"Dataset: {df.shape[0]} samples")
    
    # Try different PCA configurations
    pca_configs = [None, 15, 10, 8, 5]  # None means use all features
    
    best_yield_results = None
    best_ee_results = None
    best_yield_r2 = -float('inf')
    best_ee_r2 = -float('inf')
    
    for n_comp in pca_configs:
        print(f"\n{'='*70}")
        print(f"Testing with PCA = {n_comp if n_comp else 'All Features'}")
        print('='*70)
        
        X = match_descriptors(df, descriptors, n_components=n_comp)
        print(f"Total features: {X.shape[1]}")
        
        # Predict yield
        print("\n" + "=" * 70)
        print("Predicting YIELD")
        print("=" * 70)
        y_yield = df["yield"]
        yield_results, yield_best = train_and_evaluate_models(X, y_yield, "Yield")
        
        if yield_results[yield_best]['cv_r2'] > best_yield_r2:
            best_yield_r2 = yield_results[yield_best]['cv_r2']
            best_yield_results = yield_results
        
        # Predict ee
        print("\n" + "=" * 70)
        print("Predicting ENANTIOMERIC EXCESS (ee)")
        print("=" * 70)
        y_ee = df["ee"]
        ee_results, ee_best = train_and_evaluate_models(X, y_ee, "Enantiomeric Excess")
        
        if ee_results[ee_best]['cv_r2'] > best_ee_r2:
            best_ee_r2 = ee_results[ee_best]['cv_r2']
            best_ee_results = ee_results
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY - BEST RESULTS ACROSS ALL CONFIGURATIONS")
    print("=" * 70)
    
    print("\nYield Prediction - Best Results:")
    for name, res in best_yield_results.items():
        print(f"  {name:20s}: CV R² = {res['cv_r2']:.4f}, Test R² = {res['test_r2']:.4f}")
    
    print("\nEnantiomeric Excess Prediction - Best Results:")
    for name, res in best_ee_results.items():
        print(f"  {name:20s}: CV R² = {res['cv_r2']:.4f}, Test R² = {res['test_r2']:.4f}")
    
    # Check if target achieved
    yield_best_name = max(best_yield_results.keys(), key=lambda k: best_yield_results[k]['cv_r2'])
    ee_best_name = max(best_ee_results.keys(), key=lambda k: best_ee_results[k]['cv_r2'])
    
    yield_cv_r2 = best_yield_results[yield_best_name]['cv_r2']
    ee_cv_r2 = best_ee_results[ee_best_name]['cv_r2']
    
    print(f"\n{'='*70}")
    print(f"BEST CV R² RESULTS:")
    print(f"  Yield: {yield_best_name} - CV R² = {yield_cv_r2:.4f}")
    print(f"  ee:    {ee_best_name} - CV R² = {ee_cv_r2:.4f}")
    
    if yield_cv_r2 >= 0.6 and ee_cv_r2 >= 0.6:
        print("\n  ✅ BOTH TARGETS ACHIEVED CV R² >= 0.6!")
    elif yield_cv_r2 >= 0.6:
        print("\n  ⚠️  Only Yield achieved CV R² >= 0.6")
    elif ee_cv_r2 >= 0.6:
        print("\n  ⚠️  Only ee achieved CV R² >= 0.6")
    else:
        print(f"\n  ❌ Neither target achieved CV R² >= 0.6")
        print(f"     Yield gap: {0.6 - yield_cv_r2:.4f}")
        print(f"     ee gap:    {0.6 - ee_cv_r2:.4f}")
    
    print("=" * 70)


if __name__ == "__main__":
    main()