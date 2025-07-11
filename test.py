import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_quantile_group_cumclose(event_cache, confidence_score_list, group_num=5):
    """
    对每个test event，按第一天confidence分组，计算每组每天的 (当天close/第一天close - 1) 总和，最后对所有fold平均，画折线图
    """
    all_group_returns = []  # 每个fold: shape=(group_num, 50)
    for test_idx, test_event in enumerate(event_cache):
        confidence = confidence_score_list[test_idx]  # shape=(n_tickers, 50)
        close = test_event['close'].values  # shape=(n_tickers, 50)
        n_tickers = confidence.shape[0]
        # 分组
        first_col = confidence[:, 0]
        try:
            quantile_labels = range(group_num)
            group_idx = pd.qcut(first_col, group_num, labels=quantile_labels, duplicates='drop')
        except ValueError:
            unique_vals = np.unique(first_col)
            group_num_actual = min(group_num, len(unique_vals))
            group_idx = pd.qcut(first_col, group_num_actual, labels=range(group_num_actual), duplicates='drop')
        group_idx = np.array(group_idx)
        group_returns = np.zeros((group_num, 50))
        for g in range(group_num):
            mask = (group_idx == g)
            if np.sum(mask) == 0:
                group_returns[g, :] = np.nan
            else:
                # 计算每只股票每天的 (当天close/第一天close - 1)
                group_close = close[mask, :]
                group_cumret = (group_close / group_close[:, [0]]) - 1  # shape=(n_group_tickers, 50)
                group_returns[g, :] = np.nansum(group_cumret, axis=0)
        all_group_returns.append(group_returns)
    # 平均
    all_group_returns = np.stack(all_group_returns)  # shape=(n_folds, group_num, 50)
    mean_group_returns = np.nanmean(all_group_returns, axis=0)  # shape=(group_num, 50)
    # 画图
    x = np.arange(50)
    plt.figure(figsize=(10, 6))
    for g in range(mean_group_returns.shape[0]):
        plt.plot(x, mean_group_returns[g], label=f'Q{g+1}')
    plt.xlabel('Day')
    plt.ylabel('Sum of (close/first_close - 1)')
    plt.title('Mean Grouped Cumulative Return by Confidence Score (First Day Quantile)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    return mean_group_returns