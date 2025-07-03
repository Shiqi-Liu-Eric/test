# ------------------------------------------------------------
# 0. 依赖
# ------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tseries.offsets import BDay, MonthEnd, MonthBegin, WeekOfMonth

# ------------------------------------------------------------
# 1. 读取并整理元数据
# ------------------------------------------------------------
# 路径自行调整
df_fam   = pd.read_csv("all_fam_2123_weight.csv")
df_ftse  = pd.read_csv("ftse_all_returns_2021_2025.csv")

# -- 1.1 取 FTSE 文件里所有唯一 ticker 名称与 SEDOL 对应表
map_name_sedol = (
    df_ftse[['Name', 'SEDOL']]
    .drop_duplicates()
    .set_index('Name')
    .rename_axis('Ticker')
)

# -- 1.2 用 SEDOL 连接 fam 文件，拿到 CUSIP / ISIN / SECURITY
cols_wanted = ['SEDOL', 'CUSIP', 'ISIN', 'SECURITY']
ticker_meta = (
    map_name_sedol
    .reset_index()
    .merge(df_fam[cols_wanted].drop_duplicates(), on='SEDOL', how='left')
    .set_index('Ticker')
)

# ------------------------------------------------------------
# 2. 生成 TotRet 宽表 & ret1
# ------------------------------------------------------------
# 转宽表：行=Ticker，列=日期，值=TotRet
totret_wide = (
    df_ftse
    .pivot_table(index='Name', columns='Date', values='TotRet')
    .sort_index(axis=1)                    # 日期顺序
)

# ret1 = 日度收益率（列方向 pct_change）
ret1 = totret_wide.pct_change(axis=1)

# ------------------------------------------------------------
# 3. 构建五大指数事件日期表
# ------------------------------------------------------------
def third_friday(year, month):
    """返回 year-month 第三个星期五日期"""
    # WeekOfMonth: 3rd(week=2) Friday(day=4)
    return (pd.Timestamp(year=year, month=month, day=1)
            + WeekOfMonth(week=2, weekday=4))

def last_weekday_of_month(year, month):
    """月末最后一个工作日"""
    return (pd.Timestamp(year=year, month=month, day=1)
            + MonthEnd(1))
    
def generate_event_list(start=2021, end=2025):
    idx_names = ['FTSE', 'Russell', 'SP500', 'CRSP', 'MSCI']
    event_types = ['ranking', 'announce', 'effective']
    event_list = {idx: {tp: [] for tp in event_types} for idx in idx_names}
    
    for yr in range(start, end + 1):
        # --- FTSE (quarterly)
        for m in (3, 6, 9, 12):
            eff = third_friday(yr, m)
            rank = last_weekday_of_month(yr, m-2)  # Jan/Apr/Jul/Oct eom
            ann  = rank + BDay(5)                  # 次周一左右
            for tp, d in zip(event_types, [rank, ann, eff]):
                event_list['FTSE'][tp].append(d)
        
        # --- Russell (annual)
        rank = last_weekday_of_month(yr, 4)
        prelim = rank + pd.DateOffset(days=33)     # 6 月首周五
        eff    = last_weekday_of_month(yr, 6)
        event_list['Russell']['ranking'].append(rank)
        event_list['Russell']['announce'].append(prelim)
        event_list['Russell']['effective'].append(eff)
        
        # --- S&P 500 (quarterly share updates)
        for m in (3, 6, 9, 12):
            eff = third_friday(yr, m)
            ann = eff - BDay(5)        # 通常前周三左右
            event_list['SP500']['ranking'].append(np.nan)  # 无固定 ranking
            event_list['SP500']['announce'].append(ann)
            event_list['SP500']['effective'].append(eff)
        
        # --- CRSP (quarterly, 5-day transition)
        for m in (3, 6, 9, 12):
            start_trans = third_friday(yr, m) - BDay(2)   # Wed
            event_list['CRSP']['ranking'].append(start_trans - BDay(25))
            event_list['CRSP']['announce'].append(start_trans - BDay(10))
            event_list['CRSP']['effective'].append(start_trans)
        
        # --- MSCI (SAIR + QAIR)
        for m in (2, 5, 8, 11):
            eff = last_weekday_of_month(yr, m)
            ann = pd.Timestamp(year=yr, month=m, day=1) + MonthBegin(0) + pd.offsets.Week(1, weekday=2)
            event_list['MSCI']['ranking'].append(np.nan)
            event_list['MSCI']['announce'].append(ann)
            event_list['MSCI']['effective'].append(eff)
    
    # 转为 DataFrame / Series 方便后续
    return event_list

event_list = generate_event_list()

# ------------------------------------------------------------
# 4. 绘图函数
# ------------------------------------------------------------
def plot_ftse_return(
    df_fam=df_fam,
    ret_df=ret1,
    family_id="VANGUARD-FTSE",
    country_code=None,            # str or list
    add_del_up_down=[0, 0, 0, 0], # [Down, Up, Delete, Add]
    sub_index=None,               # str or list
    plot_period=None,             # e.g. '2023-12'
    window=20,
    country_neut=False
):
    """绘制 FTSE 加权收益曲线，并标注其他指数事件"""
    # --- 4.1 参数清理
    if country_code and isinstance(country_code, str):
        country_code = [country_code]
    if sub_index and isinstance(sub_index, str):
        sub_index = [sub_index]
    
    # --- 4.2 fam 过滤
    df = df_fam[df_fam['FamilyID'] == family_id]
    if country_code:
        df = df[df['CountryOfIssue'].isin(country_code)]
    
    change_map = {'Down Weight':0, 'Up Weight':1, 'Delete':2, 'Add':3}
    if any(add_del_up_down):
        keep_changes = [k for k,v in change_map.items() if add_del_up_down[v]]
        df = df[df['Change'].isin(keep_changes)]
    
    if sub_index:
        df = df[df['INDEX'].isin(sub_index)]
    
    if plot_period:
        df = df[df['rebal_period'] == plot_period]
    
    # --- 4.3 将 fam 的 SEDOL -> Ticker 名称
    df = df.merge(
        map_name_sedol.reset_index(),
        on='SEDOL', how='left'
    ).dropna(subset=['Name'])
    
    # --- 4.4 选定日期范围
    end_dates = pd.to_datetime(df['RebalanceTradeDate'].unique())
    if plot_period:
        end_dates = [end_dates.max()]   # 单一曲线
    # 对每个事件画一条线
    fig, ax = plt.subplots(figsize=(10,6))
    
    for ed in end_dates:
        rng = pd.date_range(ed - BDay(window), ed + BDay(window), freq='B')
        # 当天加权收益
        curve = []
        for d in rng:
            rows_today = df[df['RebalanceTradeDate'] <= d]
            w = rows_today['ProviderWeight - COB'].values
            tickers = rows_today['Name'].values
            # 收益
            r = ret_df.reindex(tickers).reindex(columns=[str(d.date())], fill_value=np.nan).values
            val = np.nansum(w * r.squeeze())
            # country neutral
            if country_neut:
                us_mask = rows_today['CountryOfIssue'] == 'United States'
                w_us = rows_today.loc[us_mask, 'ProviderWeight - COB'].values
                t_us = rows_today.loc[us_mask, 'Name'].values
                r_us = ret_df.reindex(t_us).reindex(columns=[str(d.date())], fill_value=np.nan).values
                val -= np.nansum(w_us * r_us.squeeze())
            curve.append(val)
        
        ax.plot(rng, curve, label=f'Event ending {ed.date()}')
    
    # --- 4.5 画其他指数事件竖线
    min_x, max_x = ax.get_xlim()
    all_evt_dates = []
    for idx, evts in event_list.items():
        all_evt_dates += evts['announcement'] + evts['effective']  # or include ranking
    
    for d in all_evt_dates:
        if pd.isna(d): continue
        if d >= ax.get_xlim()[0] and d <= ax.get_xlim()[1]:
            ax.axvline(d, linestyle='--', color='grey', alpha=0.5)
            # intersection 点
            if d in rng:
                y = curve[rng.get_loc(d)]
                ax.scatter(d, y, color='red', s=20, zorder=5)
    
    ax.set_title(f'FTSE weighted return ({family_id})')
    ax.set_ylabel('Return')
    ax.legend()
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------
# 用例示范（可删）
# ------------------------------------------------------------
# plot_ftse_return(family_id="VANGUARD-FTSE",
#                  country_code=['United Kingdom', 'United States'],
#                  add_del_up_down=[0,0,0,1],
#                  sub_index="FTSE ALL WORLD",
#                  plot_period="2023-12",
#                  window=20,
#                  country_neut=True)
