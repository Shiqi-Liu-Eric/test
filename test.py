import pandas as pd
import numpy as np
import random
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from collections import defaultdict
from tqdm import tqdm

random.seed(42)

# 假设 baskets 是已有的字典
basket_keys = list(baskets.keys())
random.shuffle(basket_keys)

folds = [basket_keys[i:] + basket_keys[:i] for i in range(4)]  # 每次留一个test

all_results = []
feature_importance_all = []

for fold_idx in range(4):
    test_key = basket_keys[fold_idx]
    train_keys = [k for k in basket_keys if k != test_key]
    
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list, stock_ids, date_ids = [], [], [], []

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
                    dfs["delta_weight_DMSC"].loc[ticker, feature_days[-3:]].abs().mean(),
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

    # 把train转换成np.array格式方便处理
    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)

    # 对于每个T-5 ~ T训练5个模型
    models = []
    for t in range(5):
        dtrain = xgb.DMatrix(X_train, label=y_train[:, t])
        model = xgb.train(params={"objective": "reg:squarederror"}, dtrain=dtrain, num_boost_round=100)
        models.append(model)
        feature_importance_all.append(model.get_score(importance_type='gain'))

    # 预测test集
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
                dfs["delta_weight_DMSC"].loc[ticker, feature_days[-3:]].abs().mean(),
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
            dpred = xgb.DMatrix(np.array(X_feat).reshape(1, -1))
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

# 汇总结果
result_df = pd.DataFrame(all_results)

# 画 boxplot: Winning Rate
win_box = result_df.groupby(["ticker", "T_minus"])["win"].mean().reset_index()
mape_box = result_df.groupby(["ticker", "T_minus"])["mape"].mean().reset_index()

fig, axs = plt.subplots(1, 2, figsize=(14, 6))
axs[0].boxplot([win_box[win_box["T_minus"] == i]["win"] for i in range(5, 0, -1)])
axs[0].set_title("Winning Rate Boxplot")
axs[0].set_xticklabels(["T-5", "T-4", "T-3", "T-2", "T"])
axs[0].set_ylabel("Winning Rate")

axs[1].boxplot([mape_box[mape_box["T_minus"] == i]["mape"] for i in range(5, 0, -1)])
axs[1].set_title("MAPE Boxplot")
axs[1].set_xticklabels(["T-5", "T-4", "T-3", "T-2", "T"])
axs[1].set_ylabel("MAPE")

plt.tight_layout()
plt.show()

# Feature importance 汇总
importance_df = pd.DataFrame(feature_importance_all).fillna(0)
importance_df = importance_df.groupby(importance_df.columns, axis=1).sum()
importance_df = importance_df.mean().sort_values(ascending=False).reset_index()
importance_df.columns = ["Feature", "Importance"]

plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.title("Average Feature Importance")
plt.gca().invert_yaxis()
plt.show()
