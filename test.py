import pandas as pd
import matplotlib.pyplot as plt

# 初始化用于画图的结果
acc_return_add_all = {}
acc_return_del_all = {}

# 获取所有篮子的最早和最晚日期，作为画图的横坐标范围
all_dates = []
for key, basket in baskets.items():
    all_dates.extend(basket["dates"])
min_date, max_date = min(all_dates), max(all_dates)

plot_dates = pd.date_range(min_date, max_date, freq="B")

# 初始化全局时间线上的空值
for date in plot_dates:
    acc_return_add_all[date] = None
    acc_return_del_all[date] = None

# 针对每一个 rebal event 处理
for key, basket in baskets.items():
    start_date = basket["start_date"]
    end_date = basket["end_date"]
    dates = basket["dates"]

    # 找出每一天的新增和删除成分股
    acc_return_add = []
    acc_return_del = []

    for date in dates:
        add_stocks = []
        del_stocks = []

        for ticker in tickers_list:
            code_dmss = dfs['event_code_DMSS'].loc[ticker, date] if ticker in dfs['event_code_DMSS'].index else None
            code_dmsc = dfs['event_code_DMSC'].loc[ticker, date] if ticker in dfs['event_code_DMSC'].index else None

            if code_dmss == 2 or code_dmsc == 2:
                add_stocks.append(ticker)
            elif code_dmss == -2 or code_dmsc == -2:
                del_stocks.append(ticker)

        def calc_acc_return(ticker_list):
            acc_ret = 0
            for ticker in ticker_list:
                if ticker not in dfs['close'].index or ticker not in dfs['target_weight_DMSS'].index:
                    continue
                close_series = dfs['close'].loc[ticker, dates]
                base_price = close_series[start_date]
                weighted_return = (close_series - base_price) * dfs['target_weight_DMSS'].loc[ticker, dates]
                acc_ret += weighted_return
            return acc_ret

        # 累计收益计算
        acc_add = calc_acc_return(add_stocks)
        acc_del = calc_acc_return(del_stocks)

        # 记录当前篮子在该时间段的累计收益（对每一天）
        for i, d in enumerate(dates):
            acc_return_add_all[d] = acc_add.iloc[i]
            acc_return_del_all[d] = acc_del.iloc[i]

# 转换为 Series 用于画图
acc_add_series = pd.Series(acc_return_add_all)
acc_del_series = pd.Series(acc_return_del_all)

# 画图
plt.figure(figsize=(12, 6))
acc_add_series.plot(label='Add to Proforma', linestyle='-', marker='o')
acc_del_series.plot(label='Delete from Proforma', linestyle='--', marker='x')
plt.title("Accumulate Return of Add vs Delete Stocks Over Rebalancing Periods")
plt.xlabel("Date")
plt.ylabel("Accumulate Return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
