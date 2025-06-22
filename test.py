import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime

# 初始化 Barra 类（不变）
barra = Barra()

# 日期范围
date_range = pd.date_range(start="2023-08-01", end="2023-08-28")

# 读取所有 SEDOL 代码（用于获取所有资产）
all_sedol_ids = df_id_map['SEDOL'].dropna().unique().tolist()

# 一次性读取所有 barra factors（主优化步骤）
df_all_factors = barra.get_barra_factors(
    date_range=date_range,
    factors=unique_descriptors,
    sec_ids=all_sedol_ids,
    sec_id_type='SEDOL'
)

# 设置索引加速
df_all_factors['Date'] = pd.to_datetime(df_all_factors['Date'])
df_all_factors.set_index(['Date', 'AssetID'], inplace=True)

# 初始化 dfs_new（保持原有逻辑）
tickers = dfs['ret1'].index.tolist()
dates = pd.to_datetime(dfs['ret1'].columns.tolist()[:])
dfs_new = {}

for desc in unique_descriptors:
    dfs_new[desc] = pd.DataFrame(index=tickers, columns=dates, dtype=float)
    dfs_new[desc][:] = np.nan  # 全部填 NaN

# 遍历 ticker，提取其因子（主逻辑部分）
for ticker in tqdm(tickers):
    try:
        sedol = df_id_map.loc[ticker, 'SEDOL']
        if pd.isna(sedol):
            continue

        # 提取这个 ticker 的所有行
        df_ticker = df_all_factors.xs(sedol, level='AssetID', drop_level=False)

        if df_ticker.empty:
            continue

        for desc in unique_descriptors:
            if desc in df_ticker.columns:
                valid_dates = df_ticker.index.get_level_values('Date').intersection(dfs_new[desc].columns)
                if len(valid_dates) == 0:
                    continue
                dfs_new[desc].loc[ticker, valid_dates] = df_ticker.loc[valid_dates, desc].values

    except Exception as e:
        print(f"❌ Failed for {ticker}: {e}")
        continue

# 保存为 pickle（和你原代码一致）
os.makedirs('dfs_pickle', exist_ok=True)

for desc, df in dfs_new.items():
    file_path = os.path.join('dfs_pickle', f"{desc}.pkl")
    with open(file_path, 'wb') as f:
        import pickle
        pickle.dump(df, f)

print("✅ All descriptors processed and saved.")
