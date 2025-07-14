import pandas as pd
import itertools
import operator as op

T, lag = pd.Timestamp('2021-12-15'), 50
t0_col = dfs['close'].columns[dfs['close'].columns.get_loc(T) - lag]

# 定义需要用到的运算符
ops = {
    '+': op.add,
    '-': op.sub,
    '*': op.mul,
    '/': op.truediv,          # 除零会得到 inf/NaN，稍后会 dropna
}

features = barra_names    # 你的 64 个特征名 list

close = dfs['close']
ret_50d = close[T] / close[t0_col] - 1              # 预先算好 50 日收益

rows = []                                           # 收集结果

for f1, f2 in itertools.combinations(features, 2): # 两两组合
    s1, s2 = dfs[f1][t0_col], dfs[f2][t0_col]       # 取 T-50 当天截面
    for sym, fn in ops.items():                     # 四种运算
        expr = f'{f1} {sym} {f2}'
        try:
            factor = fn(s1, s2)                     # 逐元素计算
        except Exception:
            continue                                # 极端情况跳过
        q = pd.qcut(factor, 5, labels=False, duplicates='drop')
        grp_mean = (
            pd.DataFrame({'q': q, 'r': ret_50d})
              .dropna()
              .groupby('q')['r']
              .mean()
              .reindex(range(5))                    # 确保 5 行
        )
        rows.append([expr, *grp_mean.values])

# 组装结果表：index 为表达式，列 Q1…Q5
out = pd.DataFrame(
    rows, columns=['expr', 'Q1','Q2','Q3','Q4','Q5']
).set_index('expr')

print(out)
