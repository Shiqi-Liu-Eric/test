    all_group_returns = np.stack(all_group_returns)  # shape=(n_folds, group_num, 50)
    mean_group_returns = np.nanmean(all_group_returns, axis=0)  # shape=(group_num, 50)

    # === 新增：对每组做累计收益 ===
    mean_group_cum_returns = np.cumsum(mean_group_returns, axis=1)  # shape=(group_num, 50)

    # 画图
    x = np.arange(50)
    plt.figure(figsize=(10, 6))
    for g in range(mean_group_cum_returns.shape[0]):
        plt.plot(x, mean_group_cum_returns[g], label=f'Q{g+1}')
    plt.xlabel('Day')
    plt.ylabel('Cumulative Sum of ret1')
    plt.title('Mean Grouped Cumulative ret1 by Confidence Score (First Day Quantile)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    return mean_group_cum_returns