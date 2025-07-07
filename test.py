def build_weight_matrix_v2(df_filtered, date_index):
    """
    构造每天每只股票的权重矩阵：
    - df_filtered: 已过滤后的 DataFrame，包含 ['SEDOL', 'RebalanceTradeDate', 'ProviderWeight - COB']
    - date_index: 所有交易日组成的 DatetimeIndex（与 ret1_sedol.columns 一致）
    
    返回：一个 DataFrame，index 为 SEDOL，columns 为交易日，值为权重
    """
    # 初始化矩阵为全 0
    sedols = df_filtered['SEDOL'].unique()
    w_mat = pd.DataFrame(0.0, index=sedols, columns=date_index)

    # 预处理：按 SEDOL 分组
    grouped = df_filtered[['SEDOL', 'RebalanceTradeDate', 'ProviderWeight - COB']].dropna().copy()
    grouped['RebalanceTradeDate'] = pd.to_datetime(grouped['RebalanceTradeDate'])
    grouped = grouped.sort_values(['SEDOL', 'RebalanceTradeDate'])

    for sedol, group in grouped.groupby('SEDOL'):
        dates = group['RebalanceTradeDate'].values
        weights = group['ProviderWeight - COB'].values

        for i in range(len(dates)):
            start_date = pd.to_datetime(dates[i])
            if i + 1 < len(dates):
                end_date = pd.to_datetime(dates[i + 1]) - pd.Timedelta(days=1)
            else:
                end_date = date_index[-1]  # 最后一行延续到最后一天

            # 取该 SEDOL 对应的这个区间内的所有交易日
            date_range = date_index[(date_index >= start_date) & (date_index <= end_date)]
            w_mat.loc[sedol, date_range] = weights[i]

    return w_mat
