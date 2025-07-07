import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_ftse_return(
    df_fam,
    ret1_sedol,
    event_list,
    family_id="VANGUARD-FTSE",
    country_code=None,
    add_del_up_down=[0, 0, 0, 0],
    sub_index=None,
    country_neut=False,
    event_index=None,
    event_type=None
):
    # ---------- 1. 参数清理 ----------
    if isinstance(country_code, str):
        country_code = [country_code]
    if isinstance(sub_index, str):
        sub_index = [sub_index]
    if event_index is None:
        event_index = ['FTSE', 'Russell', 'SP500', 'CRSP', 'MSCI']
    if event_type is None:
        event_type = ['ranking', 'announcement', 'effective']

    # ---------- 2. 过滤数据 ----------
    df = df_fam[df_fam['FamilyID'] == family_id]
    if country_code:
        df = df[df['CountryOfIssue'].isin(country_code)]

    change_map = {'Down Weight': 0, 'Up Weight': 1, 'Delete': 2, 'Add': 3}
    if any(add_del_up_down):
        keep = [k for k, v in change_map.items() if add_del_up_down[v]]
        df = df[df['Change'].isin(keep)]

    if sub_index:
        df = df[df['INDEX'].isin(sub_index)]

    df = df.dropna(subset=['SEDOL', 'RebalanceTradeDate', 'ProviderWeight - COB'])
    df['RebalanceTradeDate'] = pd.to_datetime(df['RebalanceTradeDate'])

    # ---------- 3. 时间范围 ----------
    all_dates = ret1_sedol.columns
    all_dates = all_dates[all_dates <= pd.Timestamp('2023-12-31')]
    ret1_sedol = ret1_sedol.loc[:, all_dates]

    # ---------- 4. 创建每天的return序列 ----------
    rebalance_dates = sorted(df['RebalanceTradeDate'].unique())
    rebalance_dates.append(all_dates[-1] + pd.Timedelta(days=1))  # 处理最后一个区间

    daily_ret = pd.Series(0.0, index=all_dates)

    for i in range(len(rebalance_dates) - 1):
        start = rebalance_dates[i]
        end = rebalance_dates[i + 1] - pd.Timedelta(days=1)
        date_range = all_dates[(all_dates >= start) & (all_dates <= end)]

        df_group = df[df['RebalanceTradeDate'] == start]

        for day in date_range:
            day_str = str(day.date())
            r_total = 0.0
            for _, row in df_group.iterrows():
                sedol = row['SEDOL']
                weight = row['ProviderWeight - COB']
                if sedol in ret1_sedol.index and day_str in ret1_sedol.columns:
                    ret = ret1_sedol.at[sedol, day_str]
                    if pd.notna(ret):
                        r_total += weight * ret
            # country neutral
            if country_neut:
                r_us = 0.0
                for _, row in df_group[df_group['CountryOfIssue'] == 'United States'].iterrows():
                    sedol = row['SEDOL']
                    weight = row['ProviderWeight - COB']
                    if sedol in ret1_sedol.index and day_str in ret1_sedol.columns:
                        ret = ret1_sedol.at[sedol, day_str]
                        if pd.notna(ret):
                            r_us += weight * ret
                r_total -= r_us
            daily_ret.loc[day] = r_total

    cum_ret = daily_ret.cumsum()

    # ---------- 5. 事件累积收益 ----------
    evt_dates = set()
    for idx in event_index:
        if idx not in event_list:
            continue
        for tp in event_type:
            evt_dates.update(event_list[idx].get(tp, []))

    evt_dates = pd.to_datetime(list(evt_dates))
    evt_dates = evt_dates[(evt_dates >= all_dates[0]) & (evt_dates <= all_dates[-1])]

    mask = daily_ret.index.isin(evt_dates)
    cum_ret_evt = (daily_ret.where(mask, 0.0)).cumsum()

    # ---------- 6. 绘图 ----------
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(daily_ret.index, daily_ret.values, lw=1.2, label='Daily total return')
    ax.set_ylabel('Daily return')
    ax.set_xlabel('Date')

    ax2 = ax.twinx()
    ax2.plot(cum_ret.index, cum_ret.values, color='tab:orange',
             label='Cumulative return (all days)', lw=1.8)
    ax2.plot(cum_ret_evt.index, cum_ret_evt.values, color='tab:green',
             label=f'Cumulative return on event days', lw=1.8, linestyle='--')
    ax2.set_ylabel('Cumulative return')

    for d in evt_dates:
        ax.axvline(d, linestyle=':', color='gray', alpha=0.3)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    plt.title(f'FTSE Strategy Return — Family: {family_id}')
    plt.tight_layout()
    plt.show()
