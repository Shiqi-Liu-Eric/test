import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# --------- 读取 SPY 收盘价 ---------
spy = pd.read_csv("spy_close.csv", parse_dates=['date'])
spy = spy[['date', 'PX_LAST']].dropna()
spy = spy.sort_values('date')
spy['spy_ret'] = spy['PX_LAST'].pct_change().shift(-1)    # 明日收益
spy['target'] = (spy['spy_ret'] > 0).astype(int)

# 日期交集
dates = pd.to_datetime(dfs['country_ftse_globalallcap'].columns)
valid_dates = sorted(set(dates) & set(spy['date']))
spy = spy[spy['date'].isin(valid_dates)].reset_index(drop=True)

# --------- 计算 US 股票掩码 ---------
country_df = dfs['country_ftse_globalallcap']
us_mask = (country_df == 'US').astype(float)              # 1 表示 US 股票，其它为 0

# --------- 构造特征 ---------
features = []
for t in range(6, len(valid_dates)-1):  # 确保有足够过去的数据
    date = valid_dates[t]
    t_idx = list(dates).index(date)

    feat1 = []  # 当日总和 - 昨日总和
    feat2 = []  # T-1~T-3总和 - T-4~T-6总和

    for f in unique_descriptors:
        df = dfs[f].fillna(0)

        cur = (df.iloc[:, t_idx] * us_mask.iloc[:, t_idx]).sum()
        prev = (df.iloc[:, t_idx-1] * us_mask.iloc[:, t_idx-1]).sum()
        feat1.append(cur - prev)

        sum_recent = sum((df.iloc[:, t_idx-i] * us_mask.iloc[:, t_idx-i]).sum() for i in range(1, 4))
        sum_past = sum((df.iloc[:, t_idx-i] * us_mask.iloc[:, t_idx-i]).sum() for i in range(4, 7))
        feat2.append(sum_recent - sum_past)

    features.append([date] + feat1 + feat2)

# --------- 构造 DataFrame ---------
columns = ['date'] + [f'{f}_diff1' for f in unique_descriptors] + [f'{f}_diff3vs6' for f in unique_descriptors]
X = pd.DataFrame(features, columns=columns)
X = X.merge(spy[['date', 'target']], on='date').dropna()

# --------- 划分训练集和测试集 ---------
X_train = X[X['date'].dt.year.isin([2021, 2022])].drop(columns=['date'])
X_test  = X[X['date'].dt.year == 2023].drop(columns=['date'])

y_train = X_train.pop('target')
y_test  = X_test.pop('target')

# --------- 训练模型 ----------
# 1. Logistic Regression
lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)
print("Logistic Regression:")
print(classification_report(y_test, lr.predict(X_test)))

# 2. XGBoost Classifier
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)
print("XGBoost Classifier:")
print(classification_report(y_test, xgb.predict(X_test)))
