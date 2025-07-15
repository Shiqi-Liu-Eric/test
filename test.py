import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# ========== 读取 SPY 收盘价 ========== #
spy = pd.read_csv("spy_close.csv", parse_dates=['date'])
spy = spy[['date', 'PX_LAST']].dropna().sort_values('date')
spy['spy_ret'] = spy['PX_LAST'].pct_change().shift(-1)
spy['target'] = (spy['spy_ret'] > 0).astype(int)

# ========== 对齐日期 ========== #
dates = pd.to_datetime(dfs['country_ftse_globalallcap'].columns)
valid_dates = sorted(set(dates) & set(spy['date']))
dates = np.array(valid_dates)
spy = spy[spy['date'].isin(valid_dates)].reset_index(drop=True)

# ========== 构造 US 掩码矩阵 ========== #
country_df = dfs['country_ftse_globalallcap'].loc[:, dates]
us_mask = (country_df == 'US').astype(float).values  # shape = (N_stocks, N_days)

# ========== 构造 Barra 特征张量 ========== #
feature_tensor = np.array([
    dfs[f].loc[:, dates].fillna(0).values * us_mask for f in unique_descriptors
])  # shape = (64, N_stocks, N_days)

# ========== 每个特征的总和（对股票维求和） ========== #
feature_sum = feature_tensor.sum(axis=1)  # shape = (64, N_days)

# ========== 构造两个特征集 ========== #
# 1. 当天 - 前一天
diff1 = feature_sum[:, 1:] - feature_sum[:, :-1]  # shape = (64, N_days - 1)

# 2. T-1~T-3 - T-4~T-6
sum_recent = feature_sum[:, 2:-3] + feature_sum[:, 3:-2] + feature_sum[:, 4:-1]
sum_past   = feature_sum[:, -6:-3] + feature_sum[:, -5:-2] + feature_sum[:, -4:-1]
diff3vs6   = sum_recent - sum_past  # shape = (64, N_days - 6)

# ========== 对齐并拼接 ========== #
min_len = min(diff1.shape[1], diff3vs6.shape[1])  # 防止越界
X_all = np.concatenate([diff1[:, -min_len:], diff3vs6[:, -min_len:]], axis=0).T  # shape = (days, 128)
X_dates = dates[6:6 + min_len]

# ========== 构造特征 DataFrame ========== #
X_df = pd.DataFrame(X_all, columns=[
    f'{f}_diff1' for f in unique_descriptors
] + [
    f'{f}_diff3vs6' for f in unique_descriptors
])
X_df['date'] = X_dates

# 合并 target（SPY）
X_df = X_df.merge(spy[['date', 'target']], on='date').dropna()

# ========== 划分训练和测试集 ========== #
X_train = X_df[X_df['date'].dt.year.isin([2021, 2022])].drop(columns=['date'])
X_test  = X_df[X_df['date'].dt.year == 2023].drop(columns=['date'])
y_train = X_train.pop('target')
y_test  = X_test.pop('target')

# ========== 模型训练和评估 ========== #
print("\n📘 Logistic Regression:")
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print(classification_report(y_test, y_pred_lr))

print("\n📗 XGBoost Classifier:")
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
print(classification_report(y_test, y_pred_xgb))
