import pandas as pd
import os

# 文件路径
input_csv = "/home/tzz/AIChem/reactionopt/benchmark/compare_mothods/edboplus/results/EDBOplus_for_B-H_HTE/merged_EDBOplus_for_B-H_HTE.csv"
hte_csv = "/home/tzz/AIChem/reactionopt/benchmark/datasets/HTE_datasets/B-H_HTE/B-H_HTE.csv"
output_dir = "/home/tzz/AIChem/reactionopt/benchmark/compare_mothods/edboplus/result"

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 读取CSV文件
df_input = pd.read_csv(input_csv, usecols=["step", "sample_index", "round_id"])
df_hte = pd.read_csv(hte_csv)

# 将step列名改为batch
df_input = df_input.rename(columns={"step": "batch"})

# HTE数据集中的所有列（包括yield和cost）
hte_cols = ["base", "ligand", "solvent", "concentration", "temperature", "yield", "cost"]

# 根据round_id分组，round_id范围是1-10，对应batch_0到batch_9
for round_id in range(1, 11):
    # 筛选当前round_id的数据
    batch_data = df_input[df_input["round_id"] == round_id].copy()

    # 删除round_id列（不需要输出）
    batch_data = batch_data.drop(columns=["round_id"])

    # 创建HTE数据列
    for col in hte_cols:
        batch_data[col] = None

    # 根据sample_index匹配HTE数据
    for idx, row in batch_data.iterrows():
        sample_index = row["sample_index"]
        # 在B-H_HTE.csv中查找对应的行
        hte_row = df_hte[df_hte["index"] == sample_index]
        if not hte_row.empty:
            for col in hte_cols:
                batch_data.at[idx, col] = hte_row.iloc[0][col]

    # 生成输出文件名 (round_id 1-10 对应 batch_0-9)
    output_file = os.path.join(output_dir, f"batch_{round_id - 1}.csv")

    # 保存到文件
    batch_data.to_csv(output_file, index=False)
    print(f"Saved {output_file} with {len(batch_data)} rows")

print("All batches have been split and saved successfully!")
