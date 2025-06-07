import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from xgboost import XGBRegressor
from sklearn.inspection import permutation_importance
import warnings

warnings.filterwarnings("ignore")

# ---------- 定义函数 ----------

def get_features_and_targets(dfs, tickers_list, trading_dates, start_date, end_date):
    # 提取区间内的数据
    date_range = [d for d in trading_dates if start_date <= d <= end_date]
    if len(date_range) <= 5:
        return None, None, None

    # 计算 rolling features
    delta_w_dmss = dfs["delta_weight_DMSS"].T[tickers_list].rolling(3).std().T
    mean_w_dmss = dfs["delta_weight_DMSS"].T[tickers_list].rolling(3).mean().T
    abs_mean_w_dmss = dfs["delta_weight_DMSS"].abs().T[tickers_list].rolling(3).mean().T

    delta_w_dmsc = dfs["delta_weight_DMSC"].T[tickers_list].rolling(3).std().T
    mean_w_dmsc = dfs["delta_weight_DMSC"].T[tickers_list].rolling(3).mean().T
    abs_mean_w_dmsc = dfs["delta_weight_DMSC"].abs().T[tickers_list].rolling(3).mean().T

    X_list, y_list, ticker_idx = [], [], []

    for d in date_range[2:-5]:  # 去掉前2天和最后5天
        for ticker in tickers_list:
            if pd.isna(dfs["EV_it"].loc[ticker, d]):
                continue
            row = [
                dfs["ret1"].loc[ticker, d],
                dfs["ret3"].loc[ticker, d],
                dfs["ret5"].loc[ticker, d],
                dfs["target_weight_DMSS"].loc[ticker, d],
                dfs["current_weight_DMSS"].loc[ticker, d],
                dfs["delta_weight_DMSS"].loc[ticker, d],
                dfs["event_code_DMSS"].loc[ticker, d],
                dfs["delta_notional_DMSS"].loc[ticker, d],
                dfs["delta_shares_DMSS"].loc[ticker, d],
                dfs["map_code"].loc[ticker, d],
                mean_w_dmss.loc[ticker, d],
                delta_w_dmss.loc[ticker, d],
                abs_mean_w_dmss.loc[ticker, d],
                dfs["target_weight_DMSC"].loc[ticker, d],
                dfs["current_weight_DMSC"].loc[ticker, d],
                dfs["delta_weight_DMSC"].loc[ticker, d],
                dfs["event_code_DMSC"].loc[ticker, d],
                dfs["delta_notional_DMSC"].loc[ticker, d],
                dfs["delta_shares_DMSC"].loc[ticker, d],
                mean_w_dmsc.loc[ticker, d],
                delta_w_dmsc.loc[ticker, d],
                abs_mean_w_dmsc.loc[ticker, d],
            ]
            if any(pd.isna(row)):
                continue
            X_list.append(row)
            y_list.append(dfs["EV_it"].loc[ticker, d])
            ticker_idx.append((ticker, d))

    X = pd.DataFrame(X_list, columns=[
        "ret1", "ret3", "ret5",
        "target_weight_DMSS", "current_weight_DMSS", "delta_weight_DMSS",
        "event_code_DMSS", "delta_notional_DMSS", "delta_shares_DMSS",
        "map_code", "mean_DMSS", "std_DMSS", "abs_mean_DMSS",
        "target_weight_DMSC", "current_weight_DMSC", "delta_weight_DMSC",
        "event_code_DMSC", "delta_notional_DMSC", "delta_shares_DMSC",
        "mean_DMSC", "std_DMSC", "abs_mean_DMSC"
    ])
    y = np.array(y_list)
    return X, y, ticker_idx


def compute_metrics(y_true, y_pred):
    win = np.mean(np.sign(y_true) == np.sign(y_pred))
    with np.errstate(divide='ignore', invalid='ignore'):
        ape = np.abs((y_true - y_pred) / y_true)
        ape[np.isinf(ape)] = 1.0  # 设置为100%
        ape = np.nan_to_num(ape, nan=1.0)
    return win, ape


# ---------- 主流程 ----------

basket_keys = list(baskets.keys())
assert len(basket_keys) == 4, "当前basket数量不是4，不能做4-fold"

results = []
all_features = []
all_importances = []

for i in range(4):
    test_key = basket_keys[i]
    train_keys = [k for k in basket_keys if k != test_key]

    # 拼接训练集
    X_train_all, y_train_all = [], []
    for k in train_keys:
        X_part, y_part, _ = get_features_and_targets(dfs, tickers_list, trading_dates,
                                                     baskets[k]['start_date'], baskets[k]['end_date'])
        if X_part is not None:
            X_train_all.append(X_part)
            y_train_all.append(y_part)

    X_train = pd.concat(X_train_all, axis=0)
    y_train = np.concatenate(y_train_all)

    # 测试集按 T-4 到 T 拆成五个模型
    X_test_all, y_test_all, idx_test_all = get_features_and_targets(dfs, tickers_list, trading_dates,
                                                                     baskets[test_key]['start_date'],
                                                                     baskets[test_key]['end_date'])
    if X_test_all is None:
        continue

    date_list = sorted(set([d for t, d in idx_test_all]))
    test_result, test_ape, test_sign = [], [], []

    for offset in range(5):
        day_target = date_list[-5 + offset]
        X_test_day = []
        y_test_day = []
        idx_day = []

        for j, (t, d) in enumerate(idx_test_all):
            if d == day_target:
                X_test_day.append(X_test_all.iloc[j])
                y_test_day.append(y_test_all[j])
                idx_day.append(t)

        if not X_test_day:
            continue

        model = XGBRegressor(n_estimators=100, max_depth=4)
        model.fit(X_train, y_train)
        y_pred = model.predict(pd.DataFrame(X_test_day, columns=X_train.columns))

        win, ape = compute_metrics(np.array(y_test_day), y_pred)
        test_result.extend([win] * len(y_test_day))
        test_ape.extend(list(ape))
        test_sign.extend(zip(idx_day, [day_target]*len(y_test_day)))

        # 特征重要性
        imp = model.feature_importances_
        all_importances.append(pd.Series(imp, index=X_train.columns))

    results.append({
        "win": test_result,
        "mape": test_ape,
        "sign_idx": test_sign
    })

# ---------- 汇总画图 ----------

# Boxplot for winning rate
all_win = [w for fold in results for w in fold["win"]]
all_mape = [m for fold in results for m in fold["mape"]]

plt.figure()
plt.boxplot(all_win)
plt.title("Winning Rate (boxplot)")
plt.ylabel("Rate")
plt.savefig("winning_rate_boxplot.png")

plt.figure()
plt.boxplot(all_mape)
plt.title("MAPE (boxplot)")
plt.ylabel("MAPE")
plt.savefig("mape_boxplot.png")

# 特征重要性
avg_importance = pd.concat(all_importances, axis=1).mean(axis=1).sort_values(ascending=False)
print("Top Feature Importances:")
print(avg_importance.head(10))

plt.figure(figsize=(10, 6))
avg_importance.head(20).plot(kind="bar")
plt.title("Top 20 Feature Importances")
plt.tight_layout()
plt.savefig("feature_importance_barplot.png")
