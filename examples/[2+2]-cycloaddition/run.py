import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


# ------------------------------
# 工具函数：用RDKit解析反应SMILES
# ------------------------------
def parse_reaction(full_rxn: str) -> tuple:
    """
    使用RDKit解析反应SMILES，返回：
    - 反应物Canonical SMILES列表
    - 产物Canonical SMILES列表
    - 转化率
    """
    if not full_rxn or ">>" not in full_rxn:
        return None

    # 步骤1：拆分「反应部分」与「转化率」（最后一个逗号后是转化率）
    last_comma_idx = full_rxn.rfind(",")
    if last_comma_idx == -1:
        return None
    reaction_smiles = full_rxn[:last_comma_idx].strip()  # 反应部分（反应物>>产物）

    reactants, _ = reaction_smiles.split(">>")

    try:
        # 步骤3：用RDKit解析反应SMILES，生成反应对象
        rxn = Chem.MolFromSmiles(reactants)
        if rxn is None:
            print(f"无法解析反应SMILES: {reaction_smiles}")
            return None, None
        # 3. 提取所有片段的Mol对象（asMols=True返回Mol列表，否则返回原子索引）
        frag_mols = Chem.GetMolFrags(rxn, asMols=True)
        # 4. 转为SMILES（可选保留立体化学：isomericSmiles=True）
        frag_smis = [Chem.MolToSmiles(frag, isomericSmiles=True) for frag in frag_mols]
        # from IPython import embed; embed(); exit()
        print(frag_smis)
        return frag_smis
    except Exception as e:
        print(f"解析反应出错: {reaction_smiles}, 错误: {e}")
        return None


# ------------------------------
# 步骤1：读取原始反应数据
# ------------------------------
raw_data = []
with open("2p2-cycloaddition.csv", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            raw_data.append({"full_rxn": line})  # 保存完整反应行

df_raw = pd.DataFrame(raw_data)

# ------------------------------
# 步骤2：解析反应，提取反应物/产物/转化率
# ------------------------------
# df_raw["reactants"] = None  # 反应物Canonical SMILES列表

for idx, row in df_raw.iterrows():
    full_rxn = row["full_rxn"]
    reactants = parse_reaction(full_rxn)
    df_raw.at[idx, "reactants"] = reactants

print(df_raw)
# ------------------------------
# 步骤3：生成待标注的唯一分子表
# ------------------------------
# 收集所有分子（反应物+产物）
all_molecules = []
for reactants in df_raw["reactants"]:
    if reactants is None:
        continue
    all_molecules.extend(reactants)

# 去重，得到唯一Canonical SMILES（标准化后的分子唯一标识）
unique_canonical_smiles = list(set(all_molecules))

# 生成待标注表（仅含Canonical SMILES和空角色列）
df_unique = pd.DataFrame({"canonical_smiles": unique_canonical_smiles, "role": ""})  # 空列，供手动标注角色（如Reactant、Catalyst）

# 保存待标注表
df_unique.to_csv("unique_molecules_to_annotate.csv", index=False)
print("✅ 待标注唯一分子表已生成：unique_molecules_to_annotate.csv")


# ------------------------------
# 步骤4：手动标注后，生成角色分明的反应
# ------------------------------
# （需先手动填写unique_molecules_to_annotate.csv的role列，保存为annotated_molecules.csv）
def generate_roles_reaction(reactants: list, products: list, role_map: dict) -> str:
    """根据角色映射，生成角色分明的反应字符串"""

    # 辅助函数：按角色分组分子
    def group_by_role(molecules: list) -> dict:
        role_groups = {}
        for smi in molecules:
            role = role_map.get(smi, "Unknown")  # 未知角色默认标记为Unknown
            role_groups.setdefault(role, []).append(smi)
        return role_groups

    # 分组反应物和产物的角色
    reactant_groups = group_by_role(reactants)
    product_groups = group_by_role(products)

    # 格式化输出（按自定义顺序排列角色，更清晰）
    role_order = ["Reactant", "Catalyst", "Ligand", "Solvent", "Additive", "Product"]
    parts = []
    for role in role_order:
        if role in reactant_groups:
            parts.append(f"{role}: {', '.join(reactant_groups[role])}")
        if role in product_groups:
            parts.append(f"{role}: {', '.join(product_groups[role])}")

    # 添加未在order中的角色（如Unknown）
    for role in reactant_groups:
        if role not in role_order:
            parts.append(f"{role}: {', '.join(reactant_groups[role])}")
    for role in product_groups:
        if role not in role_order:
            parts.append(f"{role}: {', '.join(product_groups[role])}")

    return " >> ".join(parts) if parts else "无效反应"


# 加载标注后的分子表（需手动标注后执行）
try:
    df_annotated = pd.read_csv("annotated_molecules.csv")
    role_map = dict(zip(df_annotated["canonical_smiles"], df_annotated["role"]))

    # 生成角色分明的反应
    df_raw["role_rxn"] = df_raw.apply(lambda row: generate_roles_reaction(row["reactants"], row["products"], role_map), axis=1)

    # 保存结果
    df_raw.to_csv("reactions_with_roles.csv", index=False)
    print("✅ 带角色的反应表已生成：reactions_with_roles.csv")
except FileNotFoundError:
    print("⚠️ 请先标注unique_molecules_to_annotate.csv，保存为annotated_molecules.csv后重新运行此部分")
