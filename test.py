import pandas as pd
import numpy as np
import random
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from collections import defaultdict
from tqdm import tqdm

# 假设 baskets 和 dfs 是之前预处理好的
random.seed(42)

basket_keys = list(baskets.keys())
random.shuffle(basket_keys)

folds = [basket_keys[i:] + basket_keys[:i] for i in range(4)]  # 每次留一个 test
all_results = []
feature_importance_all = []

feature_names = [
    "ret1", "ret3", "ret5",
    "target_weight_DMSS", "current_weight_DMSS", "delta_weight_DMSS", "event_code_DMSS",
    "delta_notional_DMSS", "delta_shares_DMSS", "map_code_DMSS",
    "target_weight_DMSC", "current_weight_DMSC", "delta_weight_DMSC", "event_code_DMSC",
    "delta_notional_DMSC", "delta_shares_DMSC", "map_code_DMSC",
    "tsstd_delta_weight_DMSS", "tsmean_delta_weight_DMSS", "tsmean_abs_delta_weight_DMSS",
    "tsstd_delta_weight_DMSC", "tsmean_delta_weight_DMSC", "tsmean_abs_delta_weight_DMSC"
]

for fold_idx in range(4):
    test_key = basket_keys[fold_idx]
    train_keys = [k for k in basket_keys if k != test_key]

    X_train_list, y_train_list = [], []

    for k in train_keys:
        basket = baskets[k]
        dates = sorted(basket['dates'])
        if len(dates) < 10: continue
        feature_days = dates[-10:-5]
        target_days = dates[-5:]

        for ticker in tickers_list:
            if dfs["EV_it"].loc[ticker, target_days].isnull().any(): continue
            X_feat = []
            for date in feature_days:
                f_row = [
                    dfs["ret1"].loc[ticker, date],
                    dfs["ret3"].loc[ticker, date],
                    dfs["ret5"].loc[ticker, date],
                    dfs["target_weight_DMSS"].loc[ticker, date],
                    dfs["current_weight_DMSS"].loc[ticker, date],
                    dfs["delta_weight_DMSS"].loc[ticker, date],
                    dfs["event_code_DMSS"].loc[ticker, date],
                    dfs["delta_notional_DMSS"].loc[ticker, date],
                    dfs["delta_shares_DMSS"].loc[ticker, date],
                    dfs["map_code"].loc[ticker, date],
                    dfs["target_weight_DMSC"].loc[ticker, date],
                    dfs["current_weight_DMSC"].loc[ticker, date],
                    dfs["delta_weight_DMSC"].loc[ticker, date],
                    dfs["event_code_DMSC"].loc[ticker, date],
                    dfs["delta_notional_DMSC"].loc[ticker, date],
                    dfs["delta_shares_DMSC"].loc[ticker, date],
                    dfs["map_code"].loc[ticker, date],
                    dfs["delta_weight_DMSS"].loc[ticker, feature_days[-3:]].std(),
                    dfs["delta_weight_DMSS"].loc[ticker, feature_days[-3:]].mean(),
                    dfs["delta_weight_DMSS"].loc[ticker, feature_days[-3:]].abs().mean(),
                    dfs["delta_weight_DMSC"].loc[ticker, feature_days[-3:]].std(),
                    dfs["delta_weight_DMSC"].loc[ticker, feature_days[-3:]].mean(),
                    dfs["delta_weight_DMSC"].loc[ticker, feature_days[-3:]].abs().mean()
                ]
                if any(pd.isna(f_row)):
                    X_feat = None
                    break
                X_feat.extend(f_row)
            if X_feat is None:
                continue
            y_vals = dfs["EV_it"].loc[ticker, target_days].values
            if any(pd.isna(y_vals)): continue
            X_train_list.append(X_feat)
            y_train_list.append(y_vals)

    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)

    models = []
    for t in range(5):
        dtrain = xgb.DMatrix(X_train, label=y_train[:, t], feature_names=feature_names)
        model = xgb.train(params={"objective": "reg:squarederror"}, dtrain=dtrain, num_boost_round=100)
        models.append(model)
        feature_importance_all.append(model.get_score(importance_type='gain'))

    basket = baskets[test_key]
    dates = sorted(basket['dates'])
    if len(dates) < 10: continue
    feature_days = dates[-10:-5]
    target_days = dates[-5:]

    for ticker in tickers_list:
        if dfs["EV_it"].loc[ticker, target_days].isnull().any(): continue
        X_feat = []
        for date in feature_days:
            f_row = [
                dfs["ret1"].loc[ticker, date],
                dfs["ret3"].loc[ticker, date],
                dfs["ret5"].loc[ticker, date],
                dfs["target_weight_DMSS"].loc[ticker, date],
                dfs["current_weight_DMSS"].loc[ticker, date],
                dfs["delta_weight_DMSS"].loc[ticker, date],
                dfs["event_code_DMSS"].loc[ticker, date],
                dfs["delta_notional_DMSS"].loc[ticker, date],
                dfs["delta_shares_DMSS"].loc[ticker, date],
                dfs["map_code"].loc[ticker, date],
                dfs["target_weight_DMSC"].loc[ticker, date],
                dfs["current_weight_DMSC"].loc[ticker, date],
                dfs["delta_weight_DMSC"].loc[ticker, date],
                dfs["event_code_DMSC"].loc[ticker, date],
                dfs["delta_notional_DMSC"].loc[ticker, date],
                dfs["delta_shares_DMSC"].loc[ticker, date],
                dfs["map_code"].loc[ticker, date],
                dfs["delta_weight_DMSS"].loc[ticker, feature_days[-3:]].std(),
                dfs["delta_weight_DMSS"].loc[ticker, feature_days[-3:]].mean(),
                dfs["delta_weight_DMSS"].loc[ticker, feature_days[-3:]].abs().mean(),
                dfs["delta_weight_DMSC"].loc[ticker, feature_days[-3:]].std(),
                dfs["delta_weight_DMSC"].loc[ticker, feature_days[-3:]].mean(),
                dfs["delta_weight_DMSC"].loc[ticker, feature_days[-3:]].abs().mean()
            ]
            if any(pd.isna(f_row)):
                X_feat = None
                break
            X_feat.extend(f_row)
        if X_feat is None:
            continue
        y_true = dfs["EV_it"].loc[ticker, target_days].values
        if any(pd.isna(y_true)): continue

        preds = []
        for t in range(5):
            dpred = xgb.DMatrix(np.array(X_feat).reshape(1, -1), feature_names=feature_names)
            pred = models[t].predict(dpred)[0]
            preds.append(pred)

        for t in range(5):
            pred = preds[t]
            actual = y_true[t]
            win = int((pred >= 0 and actual >= 0) or (pred < 0 and actual < 0))
            mape = 100.0 if actual == 0 else abs(pred - actual) / abs(actual)
            all_results.append({
                "ticker": ticker,
                "fold": fold_idx,
                "T_minus": 5 - t,
                "win": win,
                "mape": mape,
                "pred": pred,
                "actual": actual
            })

# 结果分析和可视化
result_df = pd.DataFrame(all_results)
win_box = result_df.groupby(["ticker", "T_minus"])["win"].mean().reset_index()
mape_box = result_df.groupby(["ticker", "T_minus"])["mape"].mean().reset_index()

# Barplot: Winning Rate
avg_win_rate = win_box.groupby("T_minus")["win"].mean().sort_index(ascending=False)
plt.figure(figsize=(8, 5))
plt.bar(["T-5", "T-4", "T-3", "T-2", "T"], avg_win_rate)
plt.title("Average Winning Rate per Day")
plt.ylabel("Winning Rate")
plt.xlabel("Rebalance Day")
plt.show()

# Boxplot: MAPE
plt.figure(figsize=(8, 5))
plt.boxplot([mape_box[mape_box["T_minus"] == i]["mape"] for i in range(5, 0, -1)])
plt.title("MAPE Boxplot")
plt.xticks([1, 2, 3, 4, 5], ["T-5", "T-4", "T-3", "T-2", "T"])
plt.ylabel("MAPE")
plt.show()

# Feature Importance Plot
importance_all = defaultdict(float)
for fmap in feature_importance_all:
    for k, v in fmap.items():
        importance_all[k] += v

importance_df = pd.DataFrame({
    "Feature": list(importance_all.keys()),
    "Importance": list(importance_all.values())
}).sort_values(by="Importance", ascending=False).head(15)

plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.title("Top 15 Feature Importances")
plt.xlabel("Gain")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
