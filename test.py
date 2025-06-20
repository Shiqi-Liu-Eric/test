import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 获取 exchange 分类
tickers = dfs['ret1'].columns.tolist()
exchange_map = {
    "UN_UW": [],
    "JT": [],
    "OTHER": []
}

for ticker in tickers:
    suffix = ticker.split()[-1]
    if suffix in ['UN', 'UW']:
        exchange_map['UN_UW'].append(ticker)
    elif suffix == 'JT':
        exchange_map['JT'].append(ticker)
    else:
        exchange_map['OTHER'].append(ticker)

# 准备 basket 列表
basket_keys = sorted(baskets.keys())

# 初始化结果字典
stat1_by_exchange = {ex: [] for ex in exchange_map}  # EV_it总和
stat2_by_exchange = {ex: [] for ex in exchange_map}  # ret1 * current_weight 总和

# 遍历每个 basket 的每一天
for key in basket_keys:
    dates = baskets[key]['dates']
    for date in dates:
        for ex in exchange_map:
            tickers_ex = exchange_map[ex]
            try:
                # 获取当天的指标
                ev_vals = dfs['EV'].loc[date, tickers_ex]
                ret_vals = dfs['ret1'].loc[date, tickers_ex]
                weight_vals = dfs['current_weight_DMSC'].loc[date, tickers_ex]

                # 清理掉 nan / inf
                valid_mask = (~ev_vals.isna()) & (~ret_vals.isna()) & (~weight_vals.isna())
                valid_mask &= np.isfinite(ev_vals) & np.isfinite(ret_vals) & np.isfinite(weight_vals)

                ev_sum = ev_vals[valid_mask].sum()
                ret_weight_sum = (ret_vals[valid_mask] * weight_vals[valid_mask]).sum()
            except:
                ev_sum = np.nan
                ret_weight_sum = np.nan

            stat1_by_exchange[ex].append(ev_sum)
            stat2_by_exchange[ex].append(ret_weight_sum)

# 将结果整理为 DataFrame，方便画图
x_labels = []
for key in basket_keys:
    x_labels.extend([d.strftime('%Y-%m-%d') for d in baskets[key]['dates']])

def plot_stats(stat_dict, title):
    fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(title, fontsize=16)

    for idx, ex in enumerate(['UN_UW', 'JT', 'OTHER']):
        axs[idx].plot(stat_dict[ex], label=ex)
        axs[idx].set_title(f"{ex}")
        axs[idx].grid(True)
        axs[idx].legend()

    axs[2].set_xticks(range(len(x_labels)))
    axs[2].set_xticklabels(x_labels, rotation=90, fontsize=6)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()

# 画图
plot_stats(stat1_by_exchange, "Statistic 1: Sum of EV_it by Exchange")
plot_stats(stat2_by_exchange, "Statistic 2: Sum of ret1 * current_weight by Exchange")
