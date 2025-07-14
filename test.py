import pandas as pd
import numpy as np

quantile_returns = {}

for T in rebalance_dates:
    T_50 = T - pd.Timedelta(days=50)
    if T_50 not in dfs["RSTR"].columns or T not in dfs["close"].columns:
        continue

    rstr_T50 = dfs["RSTR"][T_50]
    close_T50 = dfs["close"][T_50]
    close_T = dfs["close"][T]

    # 计算收益率
    ret = (close_T / close_T50 - 1).dropna()

    # 去掉RSTR或ret缺失值
    valid_idx = rstr_T50.index.intersection(ret.index)
    rstr_T50 = rstr_T50.loc[valid_idx]
    ret = ret.loc[valid_idx]

    # 分组
    quantile_labels = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
    groups = pd.qcut(rstr_T50, q=5, labels=quantile_labels)

    # 计算每组平均收益
    group_ret = ret.groupby(groups).mean()
    quantile_returns[T] = group_ret

# 汇总为DataFrame
result_df = pd.DataFrame(quantile_returns).T.sort_index()

# 添加最大组列
result_df['BestGroup'] = result_df.idxmax(axis=1)
