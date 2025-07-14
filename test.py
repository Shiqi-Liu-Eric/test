import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# -------------------- 基本配置 --------------------
T_LAG = 50                       # 50 日收益
features = barra_names           # 64 个 Barra 特征名 list
close = dfs['close']

def get_xy(date):
    """生成某一 rebalance date 的 (X, y) DataFrame"""
    t0 = close.columns[close.columns.get_loc(date) - T_LAG]
    y = close[date] / close[t0] - 1
    X = pd.concat([dfs[f][t0] for f in features], axis=1, keys=features)
    df = pd.concat([X, y.rename('ret')], axis=1).dropna()
    return df[features], df['ret']

def quintile_perf(pred, ret):
    """按预测值分 5 组求组均收益，返回长度 5 的 ndarray"""
    q = pd.qcut(pred, 5, labels=False, duplicates='drop')
    return ret.groupby(q).mean().reindex(range(5)).values

rebalance_dates = sorted(pd.to_datetime(df_fam[:, "RebalanceTradeDate"].unique()))
results = []   # (fold, model, Q1..Q5)

for test_date in rebalance_dates:                      # k-fold CV (leave-one-out on events)
    # --- 组装 train / test ---
    X_test, y_test = get_xy(test_date)
    train_dates = [d for d in rebalance_dates if d != test_date]

    X_train = pd.concat([get_xy(d)[0] for d in train_dates])
    y_train = pd.concat([get_xy(d)[1] for d in train_dates])

    # --- 1) 线性回归 ---
    lr = LinearRegression().fit(X_train, y_train)
    lr_perf = quintile_perf(pd.Series(lr.predict(X_test), index=X_test.index),
                            y_test)
    results.append([test_date, 'LR', *lr_perf])

    # --- 2) XGBoost ---
    xgb = XGBRegressor(n_estimators=300, max_depth=3,
                       learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                       objective='reg:squarederror', random_state=0)
    xgb.fit(X_train, y_train)
    xgb_perf = quintile_perf(pd.Series(xgb.predict(X_test), index=X_test.index),
                             y_test)
    results.append([test_date, 'XGB', *xgb_perf])

# -------------------- 输出 --------------------
cols = ['fold', 'model', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5']
perf_df = pd.DataFrame(results, columns=cols).set_index(['fold', 'model'])

# 各模型跨折平均表现
avg_perf = perf_df.groupby('model').mean().rename_axis('model')

print(perf_df)      # 每折分组收益
print('\n===== 平均分组收益 =====')
print(avg_perf)     # 平均分组收益
