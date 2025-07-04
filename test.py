import pandas as pd

# 第一步：读取 descriptor.csv 文件（跳过前三行，按 | 分列）
df_descript = pd.read_csv("descriptor.csv", 
                          sep='|', 
                          skiprows=3, 
                          names=['Barid', 'Descriptor', 'Value', 'DataDate', 'DescriptorType'])

# 去掉 Barid 中为空或以 '!' 开头的行
df_descript = df_descript[df_descript['Barid'].notna()]
df_descript = df_descript[~df_descript['Barid'].astype(str).str.startswith('!')]

# 第二步：读取 ticker_meta.csv（包含 SEDOL 和 Name）
df_meta = pd.read_csv("ticker_meta.csv")  # 确保这个文件包含 SEDOL 列

# 去掉 SEDOL 中的空值
df_meta = df_meta[df_meta['SEDOL'].notna()]

# 第三步：统计匹配
sedol_set = set(df_meta['SEDOL'].astype(str).unique())
barid_set = set(df_descript['Barid'].astype(str).unique())

matched = sedol_set.intersection(barid_set)

# 第四步：输出结果
print(f"共有 {len(sedol_set)} 个唯一 SEDOL")
print(f"其中有 {len(matched)} 个出现在 descriptor 文件中的 Barid 列")
print(f"匹配比例为：{len(matched) / len(sedol_set):.2%}")
