import pandas as pd


def split_descriptor_columns(input_file_path):
    """
    从输入CSV中拆分出指定前缀的列，并分别保存为新CSV

    参数:
        input_file_path (str): 原始数据集的路径（如"descriptors/dataset_B2.csv"）
    """
    # 1. 读取原始数据集
    df = pd.read_csv(input_file_path)

    print(df.shape)
    print(df.drop_duplicates().shape)
    print(df.iloc[:, 8:].drop_duplicates(keep="first").shape)

    # 2. 定义需要拆分的前缀及对应的输出文件名
    prefixes = {"solvent_": "solvent_descriptors.csv", "base_": "base_descriptors.csv", "ligand_": "ligand_descriptors.csv"}

    # 3. 遍历前缀，筛选列并保存
    for prefix, output_file in prefixes.items():
        # 筛选当前前缀的所有列（用startswith匹配）
        target_columns = [col for col in df.columns if col.startswith(prefix)]
        # 提取子DataFrame并保存（index=False：不保存行索引）
        df[target_columns].drop_duplicates().to_csv(output_file, index=False)
        print(f"已保存{prefix}前缀的列到{output_file}，共{len(target_columns)}列")


# 调用函数（替换为你的原始文件路径）
split_descriptor_columns("descriptors/dataset_B2.csv")
df = pd.read_csv("suzuki-dataset.csv")
for tp in ["solvent", "ligand", "base"]:
    df[tp].drop_duplicates().to_csv(f"rxn_space/{tp}.csv")
