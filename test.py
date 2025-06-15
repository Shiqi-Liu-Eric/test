import pandas as pd
import matplotlib.pyplot as plt

# 初始化最终结果
all_dates = pd.date_range(
    min(b["start_date"] for b in baskets.values()),
    max(b["end_date"] for b in baskets.values())
)
add_returns = pd.Series(index=all_dates, dtype=float)
del_returns = pd.Series(index=all_dates, dtype=float)

# 遍历每个basket
for basket in baskets.values():
    dates = basket["dates"]
    start, end = basket["start_date"], basket["end_date"]

    # 取当前时间段数据
    close = dfs["close"].xs(dates, level=1, drop_level=False)
    weight = dfs["target_weight_DMSS"].xs(dates, level=1, drop_level=False)
    
    # 标记add/delete股票
    event_dmss = dfs["event_code_DMSS"]
    event_dmsc = dfs.get("event_code_DMSC", pd.DataFrame()).reindex(event_dmss.index).fillna(0)
    event_code = event_dmss + event_dmsc

    add_tickers = event_code.loc[:, start].loc[lambda x: x == 2].index
    del_tickers = event_code.loc[:, start].loc[lambda x: x == -2].index

    def compute_weighted_return(tickers):
        c = close.loc[close.index.get_level_values(0).isin(tickers)]
        w = weight.loc[weight.index.get_level_values(0).isin(tickers)]
        base = c.loc[(slice(None), start), :].droplevel(1)
        ret = c.sub(base, level=0).mul(w, fill_value=0)
        daily_ret = ret.groupby(level=1).sum()
        return daily_ret

    add_ret = compute_weighted_return(add_tickers)
    del_ret = compute_weighted_return(del_tickers)

    add_returns.loc[dates] = add_returns.loc[dates].fillna(0) + add_ret
    del_returns.loc[dates] = del_returns.loc[dates].fillna(0) + del_ret

# 绘图
plt.figure(figsize=(12, 6))
plt.plot(add_returns.index, add_returns.values, label="Add to Proforma", color="green")
plt.plot(del_returns.index, del_returns.values, label="Delete from Proforma", color="red")
plt.xlabel("Date")
plt.ylabel("Accumulate Return (Weighted)")
plt.title("Add vs Delete Accumulated Return Across Baskets")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
