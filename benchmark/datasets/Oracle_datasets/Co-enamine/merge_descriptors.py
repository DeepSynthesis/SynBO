import pandas as pd
from pathlib import Path


def merge_amine_descriptors():
    # 定义文件路径
    base_dir = Path(".")
    co_enamine_path = base_dir / "Co-enamine.csv"
    cation_desc_path = base_dir / "descriptors" / "amine_ion_desc_dft.csv"
    anion_desc_path = base_dir / "descriptors" / "amine_anion_desc_dft.csv"
    output_path = base_dir / "descriptors" / "amine_merged_desc_dft.csv"

    # 读取数据
    co_enamine_df = pd.read_csv(co_enamine_path)
    cation_desc_df = pd.read_csv(cation_desc_path)
    anion_desc_df = pd.read_csv(anion_desc_path)

    # 将描述符的SMILES列设为索引以便查找
    cation_desc_dict = cation_desc_df.set_index("SMILES").to_dict("index")
    anion_desc_dict = anion_desc_df.set_index("SMILES").to_dict("index")

    # 获取描述符列名（排除SMILES列）
    cation_cols = [col for col in cation_desc_df.columns if col != "SMILES"]
    anion_cols = [col for col in anion_desc_df.columns if col != "SMILES"]

    # 为合并后的描述符创建新列名（添加前缀区分）
    cation_cols_renamed = [f"cation_{col}" for col in cation_cols]
    anion_cols_renamed = [f"anion_{col}" for col in anion_cols]

    # 存储合并后的数据
    merged_data = []

    for idx, row in co_enamine_df.iterrows():
        amine_smiles = row["amine_smiles"]

        # 分割阳离子和阴离子（按第一个点分割）
        parts = amine_smiles.split(".", 1)
        if len(parts) == 2:
            cation_smiles, anion_smiles = parts
        else:
            # 处理特殊情况：如果没有点，尝试从末尾查找
            # 或者根据SMILES特征判断（阳离子含[NH+]等，阴离子含[O-]等）
            raise ValueError(f"无法分割amine_smiles: {amine_smiles}")

        # 查找阳离子描述符
        if cation_smiles in cation_desc_dict:
            cation_values = [cation_desc_dict[cation_smiles][col] for col in cation_cols]
        else:
            raise ValueError(f"未找到阳离子描述符: {cation_smiles}")

        # 查找阴离子描述符
        if anion_smiles in anion_desc_dict:
            anion_values = [anion_desc_dict[anion_smiles][col] for col in anion_cols]
        else:
            raise ValueError(f"未找到阴离子描述符: {anion_smiles}")

        # 合并数据
        merged_row = {
            "amine_smiles": amine_smiles,
            "cation_smiles": cation_smiles,
            "anion_smiles": anion_smiles,
        }

        # 添加阳离子描述符
        for col_name, value in zip(cation_cols_renamed, cation_values):
            merged_row[col_name] = value

        # 添加阴离子描述符
        for col_name, value in zip(anion_cols_renamed, anion_values):
            merged_row[col_name] = value

        merged_data.append(merged_row)

    # 创建DataFrame并保存
    merged_df = pd.DataFrame(merged_data)
    merged_df.drop_duplicates(subset=["amine_smiles"], inplace=True)
    merged_df.drop(["cation_smiles", "anion_smiles"], axis=1, inplace=True)
    merged_df.to_csv(output_path, index=False)
    print(f"合并完成！共处理 {len(merged_df)} 条记录")
    print(f"结果保存至: {output_path}")
    print(f"总列数: {len(merged_df.columns)}")
    print(f"阳离子描述符数: {len(cation_cols)}")
    print(f"阴离子描述符数: {len(anion_cols)}")

    return merged_df


if __name__ == "__main__":
    merge_amine_descriptors()
