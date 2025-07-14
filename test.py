import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# ---------- 1. 数据准备 ----------
# ① 读取 SPY 收盘价，生成二分类 target（次日涨 =1、跌=0）
spy = pd.read_csv('spy_close.csv', parse_dates=['date'])
spy = spy.sort_values('date')
spy['ret'] = spy['PX_LAST'].pct_change()
spy['target'] = (spy['ret'].shift(-1) > 0).astype(int)           # 用“次日”走势
spy = spy[['date', 'target']].dropna()

# ② 生成每日 Barra 因子美国股票截面求和
dates = dfs['close'].columns                                       # 与因子日期对齐
us_mask = dfs['country_ftse_globalallcap'] == 'US'                 # 同形状 bool DF
sums = {f: dfs[f][us_mask].groupby(level=1).sum() for f in barra_names}
# sums[f] 的 index=sedol 删除；列=dates

# 转为 multi-column DataFrame (feature × date)
sum_df = pd.concat(sums, axis=1)           # 列层次 [feature, date]
sum_df = sum_df.T.unstack()                # index=date, columns=feature
sum_df = sum_df.sort_index()

# ---------- 2. 构造 128 维特征 ----------
delta1 = sum_df.diff()                                       # 当天 - 前一天
delta36 = sum_df.rolling(3).sum() - sum_df.shift(3).rolling(3).sum()
delta1.columns = [f'{c}_d1'  for c in delta1.columns]
delta36.columns = [f'{c}_d36' for c in delta36.columns]
X_all = pd.concat([delta1, delta36], axis=1).dropna()

# ---------- 3. 对齐 feature 与 target ----------
data = spy.merge(X_all.reset_index().rename(columns={'index': 'date'}),
                 on='date', how='inner').dropna()

train = data[(data['date'] >= '2021-01-01') & (data['date'] < '2023-01-01')]
test  = data[(data['date'] >= '2023-01-01') & (data['date'] <= '2023-12-31')]

X_train, y_train = train.drop(columns=['date', 'target']), train['target']
X_test,  y_test  = test.drop(columns=['date', 'target']),  test['target']

# ---------- 4. 训练模型 ----------
clf = XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=0,
    objective='binary:logistic'
)
clf.fit(X_train, y_train)

# ---------- 5. 评估 ----------
proba = clf.predict_proba(X_test)[:, 1]
pred  = (proba >= 0.5).astype(int)

print('Accuracy :', accuracy_score(y_test, pred))
print('AUC      :', roc_auc_score(y_test, proba))
print('\nClassification report\n', classification_report(y_test, pred))
