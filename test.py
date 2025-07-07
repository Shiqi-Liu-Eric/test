import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
#  工具：生成权重矩阵 (行=SEDOL, 列=交易日)
# ------------------------------------------------------------
def build_weight_matrix(df_filtered, date_index):
    """
    df_filtered: 已筛选后的 fam 子集，须含列
        ['SEDOL', 'RebalanceTradeDate', 'ProviderWeight - COB']
    date_index : 与 ret1_sedol.columns 相同的日期 DatetimeIndex
    """
    weight_events = (df_filtered
                     .loc[:, ['SEDOL', 'RebalanceTradeDate', 'ProviderWeight - COB']]
                     .rename(columns={'ProviderWeight - COB': 'weight'}))

    # 1) 枢轴成宽表 (行=SEDOL, 列=reb_day)
    w = (weight_events
         .drop_duplicates(['SEDOL', 'RebalanceTradeDate'])
         .pivot(index='SEDOL', columns='RebalanceTradeDate', values='weight')
         .sort_index(axis=1))

    # 2) 对所有交易日 reindex，向前填充，再把前置缺口补 0
    w = (w
         .reindex(columns=date_index, fill_value=np.nan)
         .ffill(axis=1)
         .fillna(0.0))
    return w

# ------------------------------------------------------------
#  主函数：绘图
# ------------------------------------------------------------
def plot_ftse_return(
        df_fam,            # 原 fam 表
        ret1_sedol,        # 形状 (SEDOL × date) 的日收益矩阵
        event_list,        # 前面构造好的字典
        family_id="VANGUARD-FTSE",
        country_code=None,           # str / list / None
        add_del_up_down=[0, 0, 0, 0],# [Down, Up, Del, Add]
        sub_index=None,              # str / list / None
        country_neut=False,
        event_index=None,            # list，如 ['FTSE','Russell']
        event_type=None              # list，如 ['announcement','effective']
    ):
    # ---------- 1. 参数、过滤 ----------
    if isinstance(country_code, str):
        country_code = [country_code]
    if isinstance(sub_index, str):
        sub_index = [sub_index]
    if event_index is None:
        event_index = ['FTSE', 'Russell', 'SP500', 'CRSP', 'MSCI']
    if event_type is None:
        event_type = ['ranking', 'announcement', 'effective']

    df = df_fam[df_fam['FamilyID'] == family_id]

    if country_code:
        df = df[df['CountryOfIssue'].isin(country_code)]

    change_map = {'Down Weight':0, 'Up Weight':1, 'Delete':2, 'Add':3}
    if any(add_del_up_down):
        keep = [k for k, v in change_map.items() if add_del_up_down[v]]
        df = df[df['Change'].isin(keep)]

    if sub_index:
        df = df[df['INDEX'].isin(sub_index)]

    # 如果需要 country_neut，单独记下 US sedol
    us_sedol = set(df.loc[df['CountryOfIssue'] == 'United States', 'SEDOL'])

    # ---------- 2. 时间范围 ----------
    date_index = ret1_sedol.columns
    date_index = date_index[date_index <= pd.Timestamp('2023-12-31')]  # 截到 2023 年
    ret1_sedol = ret1_sedol.loc[:, date_index]

    # ---------- 3. 权重矩阵 ----------
    w_mat = build_weight_matrix(df, date_index)

    # 让 w_mat、ret1_sedol 对齐 (行必须相同顺序)
    common_idx = w_mat.index.intersection(ret1_sedol.index)
    w_mat = w_mat.loc[common_idx]
    ret_use = ret1_sedol.loc[common_idx]

    # ---------- 4. 计算日收益 ----------
    daily_ret = (w_mat * ret_use).sum(axis=0)           # Series (date)

    if country_neut and us_sedol:
        us_idx = list(us_sedol.intersection(common_idx))
        if us_idx:  # 避免空
            daily_ret -= (w_mat.loc[us_idx] * ret_use.loc[us_idx]).sum(axis=0)

    # 累计收益
    cum_ret = daily_ret.cumsum()

    # ---------- 5. 事件蒙版 & 事件累计收益 ----------
    evt_dates = set()
    for idx_name in event_index:
        if idx_name not in event_list: continue
        for tp in event_type:
            evt_dates.update(event_list[idx_name][tp])

    evt_dates = pd.to_datetime(list(evt_dates))
    evt_dates = evt_dates[(evt_dates >= date_index[0]) & (evt_dates <= date_index[-1])]

    mask = daily_ret.index.isin(evt_dates)
    cum_ret_evt = (daily_ret.where(mask, 0)).cumsum()

    # ---------- 6. 绘图 ----------
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(daily_ret.index, daily_ret.values, lw=1.2,
            label='Daily total return')
    ax.set_ylabel('Daily return')
    ax.set_xlabel('Date')

    ax2 = ax.twinx()
    ax2.plot(cum_ret.index, cum_ret.values, color='tab:orange',
             label='Cumulative return (all days)', lw=1.8)
    ax2.plot(cum_ret_evt.index, cum_ret_evt.values, color='tab:green',
             label=f'Cumulative return on {event_index} {event_type} days', lw=1.8,
             linestyle='--')
    ax2.set_ylabel('Cumulative return')

    # 可选：在图上打竖线标记事件日
    for d in evt_dates:
        ax.axvline(d, ls=':', color='grey', alpha=0.25)

    # 合并图例
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.title(f"FTSE weighted return & cumulative P/L — {family_id}")
    plt.tight_layout()
    plt.show()
