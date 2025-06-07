import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ---- 假设已有的数据结构 ----
# dfs: dict of DataFrames，每个指标对应一个 DataFrame，index 是 ticker，columns 是 trading_dates
# baskets: dict，例如 (2022,2): {"start_date":..., "end_date":..., "dates":[...]}，dates 是实际有数据的日期

# ---- 设置 ----
basket_keys = sorted(baskets.keys())
kf = KFold(n_splits=4, shuffle=True, random_state=42)
tickers_list = dfs["EV_it"].index
results = []
feature_names = [
    "ret1", "ret3", "ret5",
    "target_weight_DMSS", "current_weight_DMSS", "delta_weight_DMSS",
    "event_code_DMSS", "delta_notional_DMSS", "delta_shares_DMSS",
    "map_code",
    "ts_std", "ts_mean", "ts_mean_abs"
]

# ---- 交叉验证 ----
for train_idx, test_idx in kf.split(basket_keys):
    train_keys = [basket_keys[i] for i in train_idx]
    test_keys = [basket_keys[i] for i in test_idx]

    for ticker in tqdm(tickers_list, desc="Per Ticker Training"):
        # 训练数据
        X_train, y_train = [], []

        for key in train_keys:
            dates = sorted(baskets[key]["dates"])[:-5]  # 剔除最后五天
            for date in dates[2:]:  # 至少留出3天用于rolling特征
                try:
                    row = [
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
                        dfs["delta_weight_DMSS"].loc[ticker].shift(1).rolling(3).std().loc[date],
                        dfs["delta_weight_DMSS"].loc[ticker].shift(1).rolling(3).mean().loc[date],
                        dfs["delta_weight_DMSS"].loc[ticker].abs().shift(1).rolling(3).mean().loc[date],
                    ]
                    if pd.isna(dfs["EV_it"].loc[ticker, date]) or any(pd.isna(row)):
                        continue
                    X_train.append(row)
                    y_train.append(dfs["EV_it"].loc[ticker, date])
                except:
                    continue

        if len(X_train) < 10:
            continue

        model = XGBRegressor(n_estimators=100, max_depth=3, random_state=42)
        model.fit(X_train, y_train)

        # 测试阶段：T-4 到 T 的预测与比较
        for key in test_keys:
            test_dates = sorted(baskets[key]["dates"])[-5:]
            for date in test_dates[2:]:  # 有rolling特征的日子
                try:
                    row = [
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
                        dfs["delta_weight_DMSS"].loc[ticker].shift(1).rolling(3).std().loc[date],
                        dfs["delta_weight_DMSS"].loc[ticker].shift(1).rolling(3).mean().loc[date],
                        dfs["delta_weight_DMSS"].loc[ticker].abs().shift(1).rolling(3).mean().loc[date],
                    ]
                    if any(pd.isna(row)) or pd.isna(dfs["EV_it"].loc[ticker, date]):
                        continue

                    y_true = dfs["EV_it"].loc[ticker, date]
                    y_pred = model.predict([row])[0]
                    win = int(np.sign(y_true) == np.sign(y_pred))
                    mape = 100 if y_true == 0 else abs((y_true - y_pred) / y_true) * 100

                    results.append({
                        "ticker": ticker,
                        "date": date,
                        "win": win,
                        "mape": mape
                    })
                except:
                    continue

# ---- 结果汇总 ----
df_result = pd.DataFrame(results)

# 1. Winning Rate Boxplot
win_df = df_result.groupby("ticker")["win"].mean().reset_index()
plt.figure()
sns.boxplot(y=win_df["win"])
plt.title("Winning Rate Boxplot")
plt.ylabel("Winning Rate")
plt.show()

# 2. MAPE Boxplot
mape_df = df_result.groupby("ticker")["mape"].mean().reset_index()
plt.figure()
sns.boxplot(y=mape_df["mape"])
plt.title("MAPE Boxplot")
plt.ylabel("MAPE (%)")
plt.show()

# 3. Feature Importance（最后一个模型）
importances = model.feature_importances_
imp_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=imp_df, x="importance", y="feature")
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()
