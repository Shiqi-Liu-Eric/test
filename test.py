import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. 读取 spy_close.csv
spy = pd.read_csv('spy_close.csv', parse_dates=['date'])
spy = spy.sort_values('date')
spy['spy_return'] = spy['PX_LAST'].pct_change()
spy['target'] = (spy['spy_return'] > 0).astype(int)
spy = spy.dropna(subset=['target'])

# 2. 获取 country_ftse_globalallcap 的所有日期
country_df = dfs['country_ftse_globalallcap']
us_dates = country_df.index

# 3. 只保留在 spy_close 和 country_ftse_globalallcap 的日期交集
common_dates = spy['date'][spy['date'].isin(us_dates)]
spy = spy[spy['date'].isin(common_dates)].reset_index(drop=True)

# 4. 特征构建
features = []
for descriptor in unique_descriptors:
    df = dfs[descriptor]
    # 只保留 US 股票
    us_df = df[df['country'] == 'US']
    # 按日期分组求和
    daily_sum = us_df.groupby(us_df.index).sum()
    # 1天差分特征
    diff1 = daily_sum - daily_sum.shift(1)
    diff1.columns = [f'{descriptor}_diff1']
    # 3天和减去4-6天和
    sum3 = daily_sum.rolling(3).sum()
    sum4_6 = daily_sum.shift(3).rolling(3).sum()
    diff3_6 = sum3 - sum4_6
    diff3_6.columns = [f'{descriptor}_diff3_6']
    # 合并
    feature_df = pd.concat([diff1, diff3_6], axis=1)
    features.append(feature_df)

# 合并所有特征
feature_all = pd.concat(features, axis=1)
feature_all = feature_all.loc[common_dates]

# 5. 对齐 target
feature_all = feature_all.reset_index()
feature_all['date'] = feature_all['index']
feature_all = feature_all.drop('index', axis=1)
data = pd.merge(feature_all, spy[['date', 'target']], on='date', how='inner')
data = data.dropna()

# 6. 划分训练/测试集
train = data[(data['date'] >= '2021-01-01') & (data['date'] < '2023-01-01')]
test = data[(data['date'] >= '2023-01-01') & (data['date'] <= '2023-12-31')]

X_train = train.drop(['date', 'target'], axis=1)
y_train = train['target']
X_test = test.drop(['date', 'target'], axis=1)
y_test = test['target']

# 7. 训练模型
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# 8. 预测与评估
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))