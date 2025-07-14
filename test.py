import pandas as pd

T = pd.Timestamp('2021-12-15')
lag = 50

close = dfs['close']
rstr  = dfs['RSTR']

# 找到 T 与 T-50 对应的列标签
t0_col = close.columns[close.columns.get_loc(T) - lag]

# 依 T-50 当日 RSTR 取分位组（5 组）
quintile = pd.qcut(rstr[t0_col], 5, labels=False)

# 计算各股票 50 日收益
ret = close[T] / close[t0_col] - 1

# 组内平均收益
group_ret = (
    pd.DataFrame({'q': quintile, 'ret': ret})
      .dropna()
      .groupby('q')['ret']
      .mean()
)

print(group_ret)
