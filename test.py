import pandas as pd

# 假设你已有以下两个 DataFrame
df_sedol = dfs['ID_SEDOL1']
df_isin = dfs['ID_ISIN']

# 初始化空结果表
result = []

for ticker in df_sedol.index:
    sedol_series = df_sedol.loc[ticker]
    isin_series = df_isin.loc[ticker]
    
    # 处理 SEDOL
    sedol_counts = sedol_series.dropna().value_counts()
    if not sedol_counts.empty:
        top_sedol = sedol_counts.idxmax()
        sedol_rate = sedol_counts.iloc[0] / sedol_counts.sum()
    else:
        top_sedol = pd.NA
        sedol_rate = pd.NA

    # 处理 ISIN
    isin_counts = isin_series.dropna().value_counts()
    if not isin_counts.empty:
        top_isin = isin_counts.idxmax()
        isin_rate = isin_counts.iloc[0] / isin_counts.sum()
    else:
        top_isin = pd.NA
        isin_rate = pd.NA

    result.append({
        'SEDOL': top_sedol,
        'ISIN': top_isin,
        'SEDOL_rate': sedol_rate,
        'ISIN_rate': isin_rate
    })

# 转为 DataFrame，以 ticker 为 index
df_id_map = pd.DataFrame(result, index=df_sedol.index)

# 可选：查看部分结果
print(df_id_map.head())
