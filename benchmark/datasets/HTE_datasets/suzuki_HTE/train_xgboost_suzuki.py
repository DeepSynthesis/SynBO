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

warnings.filterwarnings("ignore")


# 获取所有RDKit描述符名称
def get_rdkit_desc_names():
    return [desc_name for desc_name, _ in Descriptors._descList]


def process_unique_smiles_column(series, desc_names, calculator):
    """
    针对HTE数据集特性优化：提取唯一的SMILES进行计算，然后映射回原数据
    """
    unique_smiles = series.dropna().unique()
    desc_dict = {}

    # 仅对唯一的SMILES计算描述符
    for sm in unique_smiles:
        # 处理空值
        if pd.isna(sm) or str(sm).strip().lower() in ["blank_cell", "blank", "na", "nan", ""]:
            desc_dict[sm] = np.zeros(len(desc_names))
            continue

        # 如果包含~，取第一个组分
        clean_sm = str(sm).split("~")[0]

        mol = Chem.MolFromSmiles(clean_sm)
        if mol is None:
            desc_dict[sm] = np.zeros(len(desc_names))
        else:
            # 计算并存入字典
            try:
                descriptors = calculator.CalcDescriptors(mol)
                desc_dict[sm] = np.array(descriptors)
            except Exception:
                desc_dict[sm] = np.zeros(len(desc_names))

    # 映射回完整的特征矩阵
    features = []
    for sm in series:
        if pd.isna(sm):
            features.append(np.zeros(len(desc_names)))
        else:
            features.append(desc_dict.get(sm, np.zeros(len(desc_names))))

    return np.array(features)


def main():
    print("=" * 60)
    print("Suzuki HTE数据集 - XGBoost模型训练")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/6] 加载数据...")
    data_path = "suzuki_HTE.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"找不到数据集文件: {data_path}")

    df = pd.read_csv(data_path)
    print(f"数据集大小: {df.shape[0]} 条记录, {df.shape[1]} 列")

    # 目标变量
    y = df["Conversion"].values
    print(f"目标变量(Conversion)范围: {y.min():.2f} - {y.max():.2f}")

    # 2. 获取RDKit描述符名称并初始化计算器 (核心提速点)
    print("\n[2/6] 初始化RDKit描述符计算器...")
    desc_names = get_rdkit_desc_names()
    n_desc = len(desc_names)
    print(f"RDKit描述符数量: {n_desc}")

    # 只实例化一次，极大提高速度
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)

    # 3. 对5种分子类型提取描述符
    mol_types = ["solvent", "catalyst", "reactant2", "reactant1", "base"]
    print(f"\n[3/6] 提取5种分子类型的RDKit描述符: {mol_types}")

    all_features = []
    for mol_type in mol_types:
        if mol_type not in df.columns:
            print(f"  警告: {mol_type} 不在数据集中，将跳过")
            continue

        print(f"  正在极速处理 {mol_type}...")
        mol_features = process_unique_smiles_column(df[mol_type], desc_names, calculator)
        print(f"    {mol_type} 描述符形状: {mol_features.shape}")
        all_features.append(mol_features)

    # 4. 合并所有特征
    print("\n[4/6] 合并特征并处理异常值...")
    X = np.hstack(all_features)
    print(f"合并后特征矩阵形状: {X.shape}")

    # 处理RDKit可能产生的NaN和Inf值
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)

    # 5. 特征标准化
    print("\n[5/6] 特征标准化...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 6. XGBoost模型训练与自定义5-fold交叉验证
    print("\n[6/6] XGBoost模型训练与手写5-fold交叉验证...")

    # XGBoost参数网格 (针对R2 > 0.9优化)
    param_grid = [
        {"n_estimators": 300, "max_depth": 8, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 500, "max_depth": 10, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.08, "subsample": 0.85, "colsample_bytree": 0.85},
        {"n_estimators": 600, "max_depth": 8, "learning_rate": 0.03, "subsample": 0.75, "colsample_bytree": 0.75},
    ]

    best_r2 = -float("inf")
    best_params = None

    print("开始超参数搜索 (避免死锁机制)...")
    for i, params in enumerate(param_grid):
        print(f"\n测试参数组合 [{i+1}/{len(param_grid)}]:")
        print(f"  n_est={params['n_estimators']}, depth={params['max_depth']}, lr={params['learning_rate']}")

        # 定义手写交叉验证
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_scaled)):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # 每折创建新模型实例，防止内存状态污染
            model = xgb.XGBRegressor(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
                random_state=42,
                n_jobs=-1,  # 使用全部CPU核心
                verbosity=0,
            )

            # 训练模型
            model.fit(X_train, y_train)

            # 验证集预测并计算R2
            y_pred = model.predict(X_val)
            fold_r2 = r2_score(y_val, y_pred)
            fold_scores.append(fold_r2)

        mean_r2 = np.mean(fold_scores)
        std_r2 = np.std(fold_scores)

        print(f"  -> 5-fold R2: {mean_r2:.4f} (+/- {std_r2:.4f})")
        print(f"  -> 各折R2: {[f'{s:.4f}' for s in fold_scores]}")

        # 记录最佳参数
        if mean_r2 > best_r2:
            best_r2 = mean_r2
            best_params = params

    print("\n" + "=" * 60)
    print(f"✅ 搜索完成！最佳参数: {best_params}")
    print(f"🏆 最佳5-fold R2: {best_r2:.4f}")
    print("=" * 60)

    # 使用最佳参数在全部数据上训练最终模型
    print("\n使用最佳参数训练最终模型...")
    final_model = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1, verbosity=0)
    final_model.fit(X_scaled, y)

    # 计算整体训练集拟合优度
    train_r2 = r2_score(y, final_model.predict(X_scaled))
    print(f"全量数据训练集 R2: {train_r2:.4f}")

    # 保存模型
    os.makedirs("benchmark", exist_ok=True)
    model_path = "benchmark/xgboost_suzuki_model.joblib"
    scaler_path = "benchmark/xgboost_suzuki_scaler.joblib"
    joblib.dump(final_model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"模型已保存至: {model_path}")

    # 验证是否满足 > 0.9 要求
    if best_r2 > 0.9:
        print(f"\n🎉 恭喜! 成功达成目标: 5-fold R2 = {best_r2:.4f} > 0.9")
    else:
        print(f"\n⚠️ 提示: 5-fold R2 = {best_r2:.4f} < 0.9，建议尝试增加树的数量(n_estimators)或深度(max_depth)")

    return best_r2


if __name__ == "__main__":
    main()
