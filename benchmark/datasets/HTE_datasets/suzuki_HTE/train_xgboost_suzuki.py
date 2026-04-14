import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import xgboost as xgb
import joblib
import warnings
import optuna
from optuna.samplers import TPESampler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

best_parameter = {
    "n_estimators": 1274,
    "max_depth": 8,
    "learning_rate": 0.018029253904222754,
    "subsample": 0.8530597199777481,
    "colsample_bytree": 0.9505932960648681,
    "min_child_weight": 3,
    "reg_alpha": 5.091542017959273,
    "reg_lambda": 4.5283455254313336e-08,
    "gamma": 1.1504135041788988,
}


def get_rdkit_desc_names():
    return [desc_name for desc_name, _ in Descriptors._descList]


def compute_rdkit_descriptors(smiles, calculator, n_desc):
    if pd.isna(smiles) or str(smiles).strip().lower() in ["blank_cell", "blank", "na", "nan", ""]:
        return np.zeros(n_desc)

    clean_sm = str(smiles).split("~")[0]
    mol = Chem.MolFromSmiles(clean_sm)
    if mol is None:
        return np.zeros(n_desc)

    try:
        descriptors = calculator.CalcDescriptors(mol)
        return np.array(descriptors)
    except Exception:
        return np.zeros(n_desc)


def process_column_features(series, desc_names, calculator):
    unique_smiles = series.dropna().unique()
    desc_dict = {}
    n_desc = len(desc_names)

    for sm in unique_smiles:
        desc_dict[sm] = compute_rdkit_descriptors(sm, calculator, n_desc)

    features = []
    for sm in series:
        if pd.isna(sm):
            features.append(np.zeros(n_desc))
        else:
            features.append(desc_dict.get(sm, np.zeros(n_desc)))

    return np.array(features)


def load_and_prepare_data():
    """加载数据并准备特征"""
    print("=" * 70)
    print("Suzuki HTE数据集 - XGBoost + Optuna超参数优化")
    print("=" * 70)

    # 加载数据
    print("\n[1/4] 加载数据...")
    data_path = "suzuki_HTE.csv"
    df = pd.read_csv(data_path)
    print(f"数据集大小: {df.shape[0]} 条记录, {df.shape[1]} 列")

    y = df["Conversion"].values
    print(f"目标变量范围: {y.min():.2f} - {y.max():.2f}")

    # 获取RDKit描述符
    print("\n[2/4] 初始化RDKit描述符计算器...")
    desc_names = get_rdkit_desc_names()
    n_desc = len(desc_names)
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)

    # 提取特征
    mol_types = ["solvent", "ligand", "reactant2", "reactant1", "base"]
    print(f"\n[3/4] 提取RDKit描述符: {mol_types}")

    all_features = []
    for mol_type in mol_types:
        if mol_type not in df.columns:
            continue
        print(f"  处理 {mol_type}...")
        mol_features = process_column_features(df[mol_type], desc_names, calculator)
        all_features.append(mol_features)

    # 合并和预处理
    print("\n[4/4] 合并特征并预处理...")
    X = np.hstack(all_features)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # 移除零方差特征
    feature_var = np.var(X, axis=0)
    valid_features = feature_var > 1e-10
    X = X[:, valid_features]
    print(f"特征矩阵形状: {X.shape}")

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    return X_scaled, y, scaler


def objective(trial, X, y):
    """Optuna优化的目标函数"""
    # 定义超参数搜索空间
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
    }

    # 5-fold交叉验证
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    val_scores = []
    train_scores = []

    for train_idx, val_idx in kfold.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = xgb.XGBRegressor(
            **params,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            tree_method="hist",
        )

        model.fit(X_train, y_train)

        train_r2 = r2_score(y_train, model.predict(X_train))
        val_r2 = r2_score(y_val, model.predict(X_val))

        train_scores.append(train_r2)
        val_scores.append(val_r2)

    mean_val_r2 = np.mean(val_scores)
    mean_train_r2 = np.mean(train_scores)
    gap = mean_train_r2 - mean_val_r2

    # 报告中间结果
    trial.set_user_attr("train_r2", mean_train_r2)
    trial.set_user_attr("gap", gap)

    # 优化目标：最大化验证R2，但惩罚过拟合
    # 如果过拟合超过0.15，给予惩罚
    if gap > 0.15:
        return mean_val_r2 - (gap - 0.15) * 0.5

    return mean_val_r2


def main():
    # 加载数据
    X, y, scaler = load_and_prepare_data()

    print("\n" + "=" * 70)
    print("开始Optuna超参数优化...")
    print("=" * 70)

    # 创建Optuna study
    study = optuna.create_study(
        direction="maximize", sampler=TPESampler(seed=42), pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    )

    # 运行优化
    n_trials = 100
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials, show_progress_bar=True)

    # 获取最佳参数
    best_params = study.best_params
    best_r2 = study.best_value

    print("\n" + "=" * 70)
    print(f"✅ Optuna优化完成！")
    print(f"🏆 最佳5-fold R2: {best_r2:.4f}")
    print(f"📊 最佳参数: {best_params}")

    # 获取最佳trial的详细信息
    best_trial = study.best_trial
    print(f"📈 训练R2: {best_trial.user_attrs.get('train_r2', 'N/A'):.4f}")
    print(f"📉 过拟合差距: {best_trial.user_attrs.get('gap', 'N/A'):.4f}")
    print("=" * 70)

    # 使用最佳参数训练最终模型
    print("\n使用最佳参数训练最终模型...")
    final_model = xgb.XGBRegressor(
        **best_params,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        tree_method="hist",
    )
    final_model.fit(X, y)

    train_r2 = r2_score(y, final_model.predict(X))
    print(f"全量数据训练集 R2: {train_r2:.4f}")
    print(f"训练-验证差距: {train_r2 - best_r2:.4f}")

    # 保存模型
    os.makedirs("benchmark", exist_ok=True)
    joblib.dump(final_model, "benchmark/xgboost_suzuki_model.joblib")
    joblib.dump(scaler, "benchmark/xgboost_suzuki_scaler.joblib")

    # 保存最佳参数
    with open("benchmark/best_params.txt", "w") as f:
        f.write(f"Best 5-fold R2: {best_r2:.4f}\n")
        f.write(f"Best params: {best_params}\n")
    print(f"模型已保存")

    if best_r2 > 0.9:
        print(f"\n🎉 恭喜! 成功达成目标: 5-fold R2 = {best_r2:.4f} > 0.9")
    elif best_r2 > 0.88:
        print(f"\n⚠️ 当前5-fold R2 = {best_r2:.4f}")
        print("   非常接近0.9！建议增加优化迭代次数")
    else:
        print(f"\n⚠️ 当前5-fold R2 = {best_r2:.4f}")

    return best_r2


if __name__ == "__main__":
    main()
