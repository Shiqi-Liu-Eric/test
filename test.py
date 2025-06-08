import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import zscore

# --- Step 1: 找出所有 basket 中都出现且 map_code 为 1 的股票 ---

valid_tickers = []

for ticker in tickers_list:
    all_ok = True
    for k in baskets.keys():
        dates = sorted(baskets[k]["dates"])
        target_days = dates[-5:]
        vals = dfs["map_code"].loc[ticker, target_days]
        if vals.isnull().any() or not (vals == 1).all():
            all_ok = False
            break
    if all_ok:
        valid_tickers.append(ticker)

print(f"符合条件的股票数量: {len(valid_tickers)}")

# --- Step 2-4: 计算每只股票四个 basket 中 EV_it 向量的两两相关性平均值 ---

correlation_scores = []

for ticker in valid_tickers:
    ev_vectors = []
    for k in baskets.keys():
        dates = sorted(baskets[k]["dates"])
        target_days = dates[-5:]
        ev = dfs["EV_it"].loc[ticker, target_days]
        if ev.isnull().any():
            break
        ev_vectors.append(ev.values)

    if len(ev_vectors) == 4:
        corr_sum = 0
        count = 0
        for v1, v2 in combinations(ev_vectors, 2):  # 共6对
            corr = np.corrcoef(v1, v2)[0, 1]
            if not np.isnan(corr):
                corr_sum += corr
                count += 1
        if count > 0:
            correlation_scores.append(corr_sum / count)

# --- Step 5: 画 boxplot（排除异常值）+ 打印统计 ---

corr_array = np.array(correlation_scores)
filtered_corr = corr_array[(np.abs(zscore(corr_array)) < 3)]  # 去除3-sigma之外的outliers

plt.figure(figsize=(8, 5))
sns.boxplot(y=filtered_corr)
plt.title("Average Pairwise Correlation of EV_it Across Baskets (Per Stock)")
plt.ylabel("Average Pearson Correlation")
plt.grid(True)
plt.show()

# 打印基本统计信息
print("统计信息（去除outliers）:")
print(pd.Series(filtered_corr).describe())
