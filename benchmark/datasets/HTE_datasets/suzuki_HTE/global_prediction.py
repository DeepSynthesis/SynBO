"""Suzuki HTE - 全局预测：加载已保存模型进行预测 (性能优化版)"""

import os
import itertools
import numpy as np
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from tqdm import tqdm

MOL_TYPES = ["solvent", "ligand", "reactant2", "reactant1", "base"]
BATCH_SIZE = 50000  # 分批处理预测，防止内存溢出


def get_rdkit_desc_names():
    return [desc_name for desc_name, _ in Descriptors._descList]


def compute_rdkit_descriptors(smiles, calculator, n_desc):
    """计算单个SMILES的描述符，返回 float32 节省内存"""
    if pd.isna(smiles) or str(smiles).strip().lower() in ["blank_cell", "blank", "na", "nan", ""]:
        return np.zeros(n_desc, dtype=np.float32)
    mol = Chem.MolFromSmiles(str(smiles).split("~")[0])
    if mol is None:
        return np.zeros(n_desc, dtype=np.float32)
    try:
        return np.array(calculator.CalcDescriptors(mol), dtype=np.float32)
    except:
        return np.zeros(n_desc, dtype=np.float32)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("=" * 70)
    print("Suzuki HTE - 全局预测（高性能优化版）")
    print("=" * 70)

    # 1. 加载模型
    model_path = os.path.join(script_dir, "xgboost_suzuki_model_full.joblib")
    scaler_path = os.path.join(script_dir, "xgboost_suzuki_scaler_full.joblib")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print("错误: 模型文件不存在! 请先运行 cv_validation.py 训练并保存模型。")
        return

    print("加载已保存的模型...")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # 2. 加载数据与组合生成
    df = pd.read_csv("suzuki_HTE.csv")
    print(f"原始数据集: {len(df)} 条记录")

    mol_values = {mt: df[mt].dropna().unique().tolist() for mt in MOL_TYPES}
    for mt, vals in mol_values.items():
        print(f"  {mt}: {len(vals)} 个唯一值")

    # 使用集合极速查重
    original_keys = set(zip(*[df[mt] for mt in MOL_TYPES]))

    all_combinations = list(itertools.product(*[mol_values[mt] for mt in MOL_TYPES]))
    print(f"\n总组合数: {len(all_combinations)}")

    new_combinations = [c for c in all_combinations if c not in original_keys]
    print(f"需要预测的新组合数: {len(new_combinations)}")

    # 3. 预计算所有独立分子的描述符 (核心优化：先计算，缓存起来)
    print("\n预计算独立分子的特征描述符 (建立缓存)...")
    desc_names = get_rdkit_desc_names()
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
    n_desc = len(desc_names)

    # 建立全局SMILES特征缓存池 (时间复杂度从 O(组合数) 降至 O(独立分子数))
    smiles_cache = {}
    for mt in MOL_TYPES:
        for sm in mol_values[mt]:
            if sm not in smiles_cache:
                smiles_cache[sm] = compute_rdkit_descriptors(sm, calculator, n_desc)

    # 4. 计算原始数据的有效特征掩码 (valid_features)
    print("计算有效特征掩码...")
    # 预分配 Numpy 数组提高拼接速度
    total_desc_len = n_desc * len(MOL_TYPES)
    X_orig = np.empty((len(df), total_desc_len), dtype=np.float32)

    for i, row in enumerate(df.itertuples(index=False)):
        combo = [getattr(row, mt) for mt in MOL_TYPES]
        X_orig[i] = np.concatenate([smiles_cache.get(sm, np.zeros(n_desc, dtype=np.float32)) for sm in combo])

    X_orig = np.nan_to_num(X_orig, nan=0.0, posinf=0.0, neginf=0.0)
    valid_features = np.var(X_orig, axis=0) > 1e-10
    del X_orig  # 释放内存

    # 5. 对新组合进行分批组装特征与预测 (防止内存溢出)
    print("对新组合进行特征拼接与预测...")
    y_new_pred_list = []

    if len(new_combinations) > 0:
        for i in tqdm(range(0, len(new_combinations), BATCH_SIZE), desc="预测进度"):
            batch_combos = new_combinations[i : i + BATCH_SIZE]

            # 预分配当前批次的数组
            X_batch = np.empty((len(batch_combos), total_desc_len), dtype=np.float32)

            # O(1) 字典查表直接拼接
            for j, combo in enumerate(batch_combos):
                X_batch[j] = np.concatenate([smiles_cache[sm] for sm in combo])

            # 特征筛选与缩放
            X_batch = X_batch[:, valid_features]
            X_batch_scaled = scaler.transform(X_batch)
            X_batch_scaled = np.nan_to_num(X_batch_scaled, nan=0.0, posinf=0.0, neginf=0.0)

            # 预测
            y_pred_batch = model.predict(X_batch_scaled)
            y_new_pred_list.append(y_pred_batch)

        y_new_pred = np.concatenate(y_new_pred_list)
    else:
        y_new_pred = np.array([])

    # 6. 使用 Pandas 向量化构建完整数据集 (极速构建结果表)
    print("\n合并并构建最终数据集...")

    df_orig_result = df.copy()
    df_orig_result["source"] = "original"

    if len(new_combinations) > 0:
        # 基于列构造字典，比逐行 append 字典快几百倍
        new_data = {mt: [combo[j] for combo in new_combinations] for j, mt in enumerate(MOL_TYPES)}
        new_data["product"] = ""
        new_data["catalyst"] = ""
        new_data["Conversion"] = np.round(y_new_pred, 2)
        new_data["source"] = "predicted"

        df_new_result = pd.DataFrame(new_data)

        # 确保列顺序一致，合并 DataFrame
        result_df = pd.concat([df_orig_result, df_new_result], ignore_index=True)
    else:
        result_df = df_orig_result

    result_df["reaction_id"] = range(1, len(result_df) + 1)

    # 7. 整理并保存结果
    # 调整列的顺序使其更美观
    cols = ["reaction_id"] + MOL_TYPES + ["product", "catalyst", "Conversion", "source"]
    existing_cols = [c for c in cols if c in result_df.columns] + [c for c in result_df.columns if c not in cols]
    result_df = result_df[existing_cols]

    output_path = os.path.join(script_dir, "suzuki_HTE_global_prediction.csv")
    result_df.to_csv(output_path, index=False)
    print(f"结果已保存: {output_path}")

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
