import pandas as pd

# 假设你已经有 df_fam
# df_fam = pd.read_csv("your_fam_file.csv")

# 筛选出 FamilyID 为 VANGUARD-FTSE 的行
df_vanguard = df_fam[df_fam['FamilyID'] == 'VANGUARD-FTSE']

# 获取这些行中的 SEDOL（去除缺失值）
sedols_from_fam = df_vanguard['SEDOL'].dropna().unique()

# 读取外部 CSV 文件
external_df = pd.read_csv("your_external_file.csv", header=None)

# 提取第一列
column_data = external_df.iloc[:, 0]

# 提取其中含有 "SEDOL" 的字段
sedol_list = []
for row in column_data:
    parts = str(row).split('|')
    for part in parts:
        if 'SEDOL' in part:
            value = part.split('SEDOL')[1]  # 获取 SEDOL 后面的值
            value = value.strip(':| ')       # 去除多余字符
            if value:
                sedol_list.append(value)

# 去重
sedol_list = set(sedol_list)

# 匹配并计算比例
match_count = sum([1 for sedol in sedols_from_fam if sedol in sedol_list])
total_count = len(sedols_from_fam)
match_ratio = match_count / total_count if total_count > 0 else 0

print(f"匹配的 SEDOL 数量: {match_count}")
print(f"总 SEDOL 数量: {total_count}")
print(f"匹配比例: {match_ratio:.2%}")
