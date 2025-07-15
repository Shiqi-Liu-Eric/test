import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# ===== Step 1: 读取 SPY 收盘价 ===== #
spy = pd.read_csv("spy_close.csv", parse_dates=['date'])
spy = spy[['date', 'PX_LAST']].dropna().sort_values('date')
spy['spy_ret'] = spy['PX_LAST'].pct_change().shift(-1)
spy['target'] = (spy['spy_ret'] > 0).astype(int)

# ===== Step 2: 日期对齐 ===== #
dates = pd.to_datetime(dfs['country_ftse_globalallcap'].columns)
valid_dates = sorted(set(dates) & set(spy['date']))
dates = np.array(valid_dates)
spy = spy[spy['date'].isin(valid_dates)].reset_index(drop=True)

# ===== Step 3: 构造 US 掩码和 Feature 总和 ===== #
country_df = dfs['country_ftse_globalallcap'].loc[:, dates]
us_mask = (country_df == 'US').astype(float).values  # shape: (N_stocks, N_days)

# 生成 shape = (64, N_stocks, N_days)
feature_tensor = np.array([
    dfs[f].loc[:, dates].fillna(0).values * us_mask
    for f in unique_descriptors
])

# shape: (64, N_days), 每个因子每天 US 股票总和
feature_sum = feature_tensor.sum(axis=1)

# ===== Step 4: 构造 diff1 和 diff3vs6 特征 ===== #
# 使用 pandas DataFrame 构造滑动窗口
fs_df = pd.DataFrame(
    feature_sum.T, index=dates, columns=unique_descriptors
)

# diff1 = 当天 - 昨天
diff1_df = fs_df.diff()

# diff3vs6 = 2 × T-1~T-3 总和 − T-1~T-6 总和
sum_1_3 = fs_df.rolling(3).sum().shift(1)
sum_1_6 = fs_df.rolling(6).sum().shift(1)
diff3vs6_df = 2 * sum_1_3 - sum_1_6

# 只保留两者都有值的日期（第 6 天以后）
valid_idx = diff3vs6_df.index[5:]

feat_df = pd.concat(
    [diff1_df.loc[valid_idx], diff3vs6_df.loc[valid_idx]],
    axis=1,
    keys=['diff1', 'diff3vs6']
).dropna()

# 列名展平为 diff1_factorA, diff3vs6_factorA, ...
feat_df.columns = [
    f'{grp}_{col}' for grp, col in feat_df.columns.to_flat_index()
]

# ===== Step 5: 合并 SPY target，准备建模数据 ===== #
X_df = (
    feat_df.reset_index()
           .merge(spy[['date', 'target']], left_on='index', right_on='date')
           .drop(columns=['index'])
           .dropna()
)

# ===== Step 6: 划分训练集和测试集（2023 年为测试） ===== #
X_train = X_df[X_df['date'].dt.year.isin([2021, 2022])].drop(columns=['date'])
X_test  = X_df[X_df['date'].dt.year == 2023].drop(columns=['date'])
y_train = X_train.pop('target')
y_test  = X_test.pop('target')

# ===== Step 7: 模型训练与评估 ===== #
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
