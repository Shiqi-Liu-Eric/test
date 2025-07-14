import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# -------------------- 配置 --------------------
T_LAG = 49
features = barra_names
close = dfs['close']

def get_xy(date):
    t0 = close.columns[close.columns.get_loc(date) - T_LAG]
    y = close[date] / close[t0] - 1
    X = pd.concat([dfs[f][t0] for f in features], axis=1, keys=features)
    df = pd.concat([X, y.rename('ret')], axis=1).dropna()
    return df[features], df['ret']

def quintile_perf(pred, ret):
    q = pd.qcut(pred, 5, labels=False, duplicates='drop')
    return ret.groupby(q).mean().reindex(range(5)).values

results = []

for test_date in rebalance_dates:
    X_test, y_test = get_xy(test_date)
    train_dates = [d for d in rebalance_dates if d != test_date]
    X_train = pd.concat([get_xy(d)[0] for d in train_dates])
    y_train = pd.concat([get_xy(d)[1] for d in train_dates])

    # ------- Feature selection -------
    lr_temp = LinearRegression().fit(X_train, y_train)
    coef_series = pd.Series(np.abs(lr_temp.coef_), index=X_train.columns)
    top10 = coef_series.nlargest(10).index.tolist()

    X_train_sel, X_test_sel = X_train[top10], X_test[top10]

    # ------- 1. 线性回归 -------
    lr = LinearRegression().fit(X_train_sel, y_train)
    pred_lr = pd.Series(lr.predict(X_test_sel), index=X_test_sel.index)
    r2_lr = r2_score(y_test, pred_lr)
    perf_lr = quintile_perf(pred_lr, y_test)
    print(f"[LR] Fold: {test_date.date()}, R²: {r2_lr:.4f}, Top features: {top10}")
    results.append([test_date, 'LR', r2_lr, *perf_lr])

    # ------- 2. XGBoost -------
    xgb = XGBRegressor(n_estimators=300, max_depth=3,
                       learning_rate=0.05, subsample=0.8,
                       colsample_bytree=0.8, objective='reg:squarederror',
                       random_state=0)
    xgb.fit(X_train_sel, y_train)
    pred_xgb = pd.Series(xgb.predict(X_test_sel), index=X_test_sel.index)
    r2_xgb = r2_score(y_test, pred_xgb)
    perf_xgb = quintile_perf(pred_xgb, y_test)
    print(f"[XGB] Fold: {test_date.date()}, R²: {r2_xgb:.4f}, Top features: {top10}")
    results.append([test_date, 'XGB', r2_xgb, *perf_xgb])

# -------------------- 输出 --------------------
cols = ['fold', 'model', 'r2', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5']
perf_df = pd.DataFrame(results, columns=cols).set_index(['fold', 'model'])

avg_perf = perf_df.groupby('model').mean().rename_axis('model')

print("\n===== 每折表现 =====")
print(perf_df)

print("\n===== 平均表现（含R²）=====")
print(avg_perf)
