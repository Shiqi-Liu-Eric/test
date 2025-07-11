import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_quantile_group_returns(event_cache, confidence_score_list, group_num=5):
    """
    event_cache: list，每个元素为dict，包含'tickers', 'ret1' (DataFrame, shape=(n_tickers, 50))
    confidence_score_list: list，每个元素为shape=(n_tickers, 50)的ndarray
    group_num: 分组数
    """
    all_group_returns = []  # 每个fold: shape=(group_num, 50)
    for test_idx, test_event in enumerate(event_cache):
        confidence = confidence_score_list[test_idx]  # shape=(n_tickers, 50)
        ret1 = test_event['ret1'].values  # shape=(n_tickers, 50)
        n_tickers = confidence.shape[0]
        # 分组
        first_col = confidence[:, 0]
        # pandas qcut 自动处理重复值
        try:
            quantile_labels = range(group_num)
            group_idx = pd.qcut(first_col, group_num, labels=quantile_labels, duplicates='drop')
        except ValueError:
            # 如果分组不够，降级为实际分组数
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
                group_returns[g, :] = np.nansum(ret1[mask, :], axis=0)
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
    plt.ylabel('Sum of ret1')
    plt.title('Mean Grouped ret1 by Confidence Score (First Day Quantile)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    return mean_group_returns

# 用法示例
# confidence_score_list = [每个test的confidence_matrix, shape=(n_tickers, 50)]
# plot_quantile_group_returns(event_cache, confidence_score_list, group_num=5)