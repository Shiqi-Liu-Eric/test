import pandas as pd
import numpy as np
from tqdm import tqdm

# ------------------------------------------
# 你已有的对象 / 函数
# ------------------------------------------
# all_tickers        : list[str]  —— 8021 只股票，索引与 shape_base 行一致
# trading_dates      : list[pd.Timestamp]
# shape_base         : pd.DataFrame，同 shape 的空框架 (index=all_tickers, columns=trading_dates)
# get_market_data(...) : 你自己写的函数，返回逐笔盘口 DataFrame
#   其列必须至少包含：
#     ListingId, Ticker, BidQuantity1~5, AskQuantity1~5, EventTimestamp
# ------------------------------------------

# ① 先准备 12 张空框架
features = [
    "VolAsk5_sum", "VolBid5_sum", "VolImb_sum", "VolImb_stddev",
    "VolAsk1_All", "VolBid1_All",
    "VolRowCount_mean", "VolRowCount_stddev",
    "VolAsk1_2345_mean", "VolAsk1_2345_stddev",
    "VolBid1_2345_mean", "VolBid1_2345_stddev"
]
raw_feature_dfs = {f: shape_base.copy(deep=True) * np.nan for f in features}

# ② 分批抓 ticker，避免一次性请求过大
batch_size = 200
ticker_batches = [all_tickers[i:i + batch_size] for i in range(0, len(all_tickers), batch_size)]

# ------------------------------------------
# ③ 主循环：date × ticker_batch
# ------------------------------------------
for date in tqdm(trading_dates, desc="dates"):
    date_str = pd.to_datetime(date).strftime("%Y-%m-%d")
    print(f"\n📅 Processing {date_str}")
    
    for batch in ticker_batches:
        # 部分 SEDOL 末尾带 “UN/ UW” 的做法，如果你想截尾就保留下面一行
        tickers = [tic[:-3] for tic in batch]   # 例如 “AAPL UW” → “AAPL”
        
        # ---------- 抓数据 ----------
        try:
            df = get_market_data(
                "XNAS",          # 也可以换成 XNYS 再跑一次后做 merge
                date_str,
                "12",
                tickers=tickers,
                columns=[
                    "ListingId", "Ticker",
                    "BidQuantity1", "BidQuantity2", "BidQuantity3", "BidQuantity4", "BidQuantity5",
                    "AskQuantity1", "AskQuantity2", "AskQuantity3", "AskQuantity4", "AskQuantity5",
                    "EventTimestamp"
                ]
            )
        except Exception as e:
            print(f"❌ {date_str} – batch 抓数失败：{e}")
            continue
        
        if df.empty:
            continue
        
        # ---------- 列运算 ----------
        bid_cols = [f"BidQuantity{i}" for i in range(1, 6)]
        ask_cols = [f"AskQuantity{i}" for i in range(1, 6)]
        
        df["VolBid5"] = df[bid_cols].sum(axis=1)
        df["VolAsk5"] = df[ask_cols].sum(axis=1)
        denom = (df["VolBid5"] + df["VolAsk5"]).replace(0, np.nan)
        df["VolImb"] = (df["VolBid5"] - df["VolAsk5"]) / denom
        
        # Level-1 单档占比
        df["Ask1_only"] = (df["AskQuantity1"] > 0) & (df[ask_cols[1:]].sum(axis=1) == 0)
        df["Bid1_only"] = (df["BidQuantity1"] > 0) & (df[bid_cols[1:]].sum(axis=1) == 0)
        
        # Ask1 / Ask(1~5)、Bid1 / Bid(1~5)
        df["Ask1_ratio"] = df["AskQuantity1"] / df[ask_cols].sum(axis=1).replace(0, np.nan)
        df["Bid1_ratio"] = df["BidQuantity1"] / df[bid_cols].sum(axis=1).replace(0, np.nan)
        
        # ---------- 按分钟聚合 ----------
        df["Minute"] = pd.to_datetime(df["EventTimestamp"]).dt.floor("min")
        grp = df.groupby("Minute")
        
        # a) 行数
        row_counts = grp.size()
        VolRowCount_mean    = row_counts.mean()
        VolRowCount_stddev  = row_counts.std()
        
        # b) Ask1_2345 / Bid1_2345
        ask1_ratio_minute = grp["Ask1_ratio"].mean()
        bid1_ratio_minute = grp["Bid1_ratio"].mean()
        VolAsk1_2345_mean    = ask1_ratio_minute.mean()
        VolAsk1_2345_stddev  = ask1_ratio_minute.std()
        VolBid1_2345_mean    = bid1_ratio_minute.mean()
        VolBid1_2345_stddev  = bid1_ratio_minute.std()
        
        # ---------- 整日汇总 ----------
        VolAsk5_sum     = df["VolAsk5"].sum()
        VolBid5_sum     = df["VolBid5"].sum()
        VolImb_sum      = df["VolImb"].sum()
        VolImb_stddev   = df["VolImb"].std()
        VolAsk1_All     = df["Ask1_only"].mean()   # 比例
        VolBid1_All     = df["Bid1_only"].mean()
        
        # ---------- 写回 12 张表 ----------
        values = [
            VolAsk5_sum, VolBid5_sum, VolImb_sum, VolImb_stddev,
            VolAsk1_All, VolBid1_All,
            VolRowCount_mean, VolRowCount_stddev,
            VolAsk1_2345_mean, VolAsk1_2345_stddev,
            VolBid1_2345_mean, VolBid1_2345_stddev
        ]
        
        for f, v in zip(features, values):
            # 对同一批次里的所有 ticker 都写入同一个值
            raw_feature_dfs[f].loc[batch, date] = v

print("✅ 全部日期处理完毕！")
