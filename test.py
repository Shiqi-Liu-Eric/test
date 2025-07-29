def calc_quantile_returns(notional_matrix, acc_return, n_quantiles=5):
    """
    计算非0 position 按 notional 分组的 return 之和
    """
    # 获取非0位置
    non_zero_mask = (notional_matrix != 0) & (~np.isnan(notional_matrix)) & (~np.isnan(acc_return))
    non_zero_notional = notional_matrix[non_zero_mask]
    non_zero_returns = acc_return[non_zero_mask]

    if len(non_zero_notional) == 0:
        return np.zeros(n_quantiles)

    # 按 notional 值分组（必须是一维）
    quantiles = pd.qcut(non_zero_notional.to_numpy().ravel(), n_quantiles, labels=False, duplicates='drop')

    quantile_returns = []
    for q in range(n_quantiles):
        mask = quantiles == q
        if np.sum(mask) > 0:
            quantile_return = np.sum(non_zero_returns.to_numpy().ravel()[mask])
        else:
            quantile_return = 0
        quantile_returns.append(quantile_return)

    return np.array(quantile_returns)
