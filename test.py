import pandas as pd
import os
import pickle
from tqdm import tqdm
import numpy as np

# === Step 1: 提取 unique descriptor 名称 ===
descriptor_file = 'EFMGEMTR_100_Asset_Std_Descript.csv'  # 替换为你自己的路径
df_desc = pd.read_csv(descriptor_file, sep='|', header=3, names=["BarraID", "Descriptor", "Value", "DataDate", "DescriptorType"])
unique_descriptors = df_desc['Descriptor'].dropna().unique().tolist()

# === Step 2: 使用已有 dfs['ret1'] 的结构初始化所有 descriptor matrix ===
ret1 = dfs['ret1']  # 你已有的矩阵
tickers = ret1.index.tolist()
dates = pd.to_datetime(ret1.columns.tolist())  # 保证是 datetime 类型

# 初始化所有 descriptor 的 dataframe，值全为 NaN
dfs_new = {}
for desc in unique_descriptors:
    dfs_new[desc] = pd.DataFrame(index=tickers, columns=dates, dtype=float)
    dfs_new[desc][:] = np.nan  # 显式填为 nan，确保全是 nan 而不是空对象

# === Step 3: 逐个 ticker 读取暴露值并写入 ===
barra = Barra()  # 假设你已经有这个类

for ticker in tqdm(tickers):
    try:
        df_factors = barra.get_barra_factors(
            date_range=dates,
            selected_factors=unique_descriptors,
            sec_ids=[ticker],
            sec_id_type='CUSIP'  # or SEDOL
        )

        # 如果成功获取
        if not df_factors.empty:
            df_factors['Date'] = pd.to_datetime(df_factors['Date'])
            df_factors.set_index('Date', inplace=True)

            for desc in unique_descriptors:
                if desc in df_factors.columns:
                    # intersection of available dates in df_factors and target dates
                    valid_dates = df_factors.index.intersection(dfs_new[desc].columns)
                    dfs_new[desc].loc[ticker, valid_dates] = df_factors.loc[valid_dates, desc]
    except Exception as e:
        print(f"❌ Failed for {ticker}: {e}")
        continue

# === Step 4: 保存为 pickle 文件 ===
os.makedirs('dfs_pickle', exist_ok=True)

for desc, df in dfs_new.items():
    file_path = os.path.join('dfs_pickle', f"{desc}.pkl")
    with open(file_path, 'wb') as f:
        pickle.dump(df, f)

print("✅ All descriptors processed and saved.")
