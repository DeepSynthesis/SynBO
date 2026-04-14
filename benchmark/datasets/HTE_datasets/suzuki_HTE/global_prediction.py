"""Suzuki HTE - 全局预测：加载已保存模型进行预测"""

import os, itertools, numpy as np, pandas as pd, joblib
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import warnings

warnings.filterwarnings("ignore")

MOL_TYPES = ["solvent", "ligand", "reactant2", "reactant1", "base"]


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


def build_feature_vector(mols_dict, mol_types, desc_names, calculator):
    n_desc = len(desc_names)
    features = []
    for mt in mol_types:
        sm = mols_dict.get(mt, None)
        if sm is None:
            features.append(np.zeros(n_desc))
        else:
            features.append(compute_rdkit_descriptors(sm, calculator, n_desc))
    return np.hstack(features)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("=" * 70)
    print("Suzuki HTE - 全局预测（所有分子组合）")
    print("=" * 70)

    # 加载已保存的模型和scaler
    model_path = os.path.join(script_dir, "xgboost_suzuki_model_full.joblib")
    scaler_path = os.path.join(script_dir, "xgboost_suzuki_scaler_full.joblib")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("错误: 模型文件不存在! 请先运行 cv_validation.py 训练并保存模型。")
        return

    print("加载已保存的模型...")
    model = joblib.load(model_path)
    valid_features = joblib.load(os.path.join(script_dir, "valid_features.joblib"))
    scaler = joblib.load(scaler_path)
    print("模型加载成功!")

    # 加载原始数据
    df = pd.read_csv("suzuki_HTE.csv")
    print(f"原始数据集: {len(df)} 条记录")

    # 获取各分子类型的唯一值
    mol_values = {mt: df[mt].dropna().unique().tolist() for mt in MOL_TYPES}
    for mt, vals in mol_values.items():
        print(f"  {mt}: {len(vals)} 个唯一值")

    # 创建原始数据的key集合
    original_keys = set()
    for _, row in df.iterrows():
        key = tuple(row[mt] for mt in MOL_TYPES)
        original_keys.add(key)

    # 生成所有可能的组合
    all_combinations = list(itertools.product(*[mol_values[mt] for mt in MOL_TYPES]))
    print(f"\n总组合数: {len(all_combinations)}")
    print(f"原始数据中的组合数: {len(original_keys)}")
    new_combinations = [c for c in all_combinations if c not in original_keys]
    print(f"需要预测的新组合数: {len(new_combinations)}")

    # 计算特征
    print("\n准备特征...")
    desc_names = get_rdkit_desc_names()
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)

    # 计算新组合的特征并预测
    print("对新组合进行预测...")
    X_new = []
    for combo in new_combinations:
        mols_dict = dict(zip(MOL_TYPES, combo))
        feat = build_feature_vector(mols_dict, MOL_TYPES, desc_names, calculator)
        X_new.append(feat)

    if len(X_new) > 0:
        X_new = np.array(X_new)[:, valid_features]
        X_new_scaled = scaler.transform(X_new)
        X_new_scaled = np.nan_to_num(X_new_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        y_new_pred = model.predict(X_new_scaled)
        y_new_pred = np.clip(y_new_pred, 0, 100)
        y_new_pred = np.round(y_new_pred, 2)
    else:
        y_new_pred = np.array([])

    # 构建完整数据集
    print("\n构建完整数据集...")
    result_rows = []
    for _, row in df.iterrows():
        result_rows.append(
            {
                **{mt: row[mt] for mt in MOL_TYPES},
                "product": row.get("product", ""),
                "catalyst": row.get("catalyst", ""),
                "Conversion": row["Conversion"],
                "source": "original",
            }
        )

    for i, combo in enumerate(new_combinations):
        result_rows.append(
            {
                **{mt: combo[j] for j, mt in enumerate(MOL_TYPES)},
                "product": "",
                "catalyst": "",
                "Conversion": round(y_new_pred[i], 2),
                "source": "predicted",
            }
        )

    result_df = pd.DataFrame(result_rows)
    result_df["reaction_id"] = range(1, len(result_df) + 1)

    # 保存结果
    output_path = os.path.join(script_dir, "suzuki_HTE_global_prediction.csv")
    result_df.to_csv(output_path, index=False)
    print(f"\n结果已保存: {output_path}")

    # 统计信息
    print("\n" + "=" * 50)
    print("统计信息:")
    print(f"  原始数据: {len(df)} 条")
    print(f"  预测数据: {len(new_combinations)} 条")
    print(f"  总计: {len(result_df)} 条")
    if len(y_new_pred) > 0:
        print(f"\n预测值统计:")
        print(f"  Min: {y_new_pred.min():.2f}")
        print(f"  Max: {y_new_pred.max():.2f}")
        print(f"  Mean: {y_new_pred.mean():.2f}")
        print(f"  Std: {y_new_pred.std():.2f}")
    print("\n✅ 全局预测完成!")


if __name__ == "__main__":
    main()
