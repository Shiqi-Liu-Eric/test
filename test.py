# ---------- 4. 创建每天的return序列（矢量化版本） ----------

# 初始化返回值
daily_ret = pd.Series(0.0, index=all_dates)

# 准备：所有交易日
rebalance_dates = sorted(df['RebalanceTradeDate'].unique())
rebalance_dates.append(all_dates[-1] + pd.Timedelta(days=1))  # 加上末尾

# 所有子表拼起来
df_all = []

for i in range(len(rebalance_dates) - 1):
    start = rebalance_dates[i]
    end   = rebalance_dates[i + 1] - pd.Timedelta(days=1)
    days  = all_dates[(all_dates >= start) & (all_dates <= end)]

    df_group = df[df['RebalanceTradeDate'] == start].copy()
    if df_group.empty or days.empty:
        continue

    # 为每一行生成对应的 Date 列（repeat + tile 方式展开）
    repeated = pd.DataFrame({
        'SEDOL': np.repeat(df_group['SEDOL'].values, len(days)),
        'Weight': np.repeat(df_group['ProviderWeight - COB'].values, len(days)),
        'Country': np.repeat(df_group['CountryOfIssue'].values, len(days)),
        'Date': np.tile(days, len(df_group))
    })

    df_all.append(repeated)

# 拼接所有重平衡区间
df_all = pd.concat(df_all, ignore_index=True)

# 确保 Date 为字符串列以对应 ret1_sedol.columns
df_all['Date'] = pd.to_datetime(df_all['Date'])
df_all = df_all[df_all['Date'] <= pd.Timestamp("2023-12-31")]

# 对应 ret 值
ret_lookup = ret1_sedol.stack().rename("Return").reset_index()
ret_lookup.columns = ['SEDOL', 'Date', 'Return']
ret_lookup['Date'] = pd.to_datetime(ret_lookup['Date'])

# Merge 得到每行 return
df_all = df_all.merge(ret_lookup, on=['SEDOL', 'Date'], how='left')

# 可选：做 country neutral，先保存 US 的贡献
if country_neut:
    us_ret = (
        df_all[df_all['Country'] == 'United States']
        .assign(Prod=lambda x: x['Weight'] * x['Return'])
        .groupby('Date')['Prod'].sum()
    )

# 总收益
df_all['Prod'] = df_all['Weight'] * df_all['Return']
daily_ret = df_all.groupby('Date')['Prod'].sum().reindex(all_dates, fill_value=0.0)

# 如果是 country_neut，需要减去 US 部分
if country_neut:
    daily_ret = daily_ret - us_ret.reindex(all_dates, fill_value=0.0)

# 累积
cum_ret = daily_ret.cumsum()
