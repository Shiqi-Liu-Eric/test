import pandas as pd
import numpy as np
import polars as pl
from bmll import get_market_data

# 假设 ret1 和 df_id_map 已经加载
all_tickers = list(df_id_map.index)
shape_base = ret1.copy(deep=False) * np.nan
date_list = ret1.columns

features = [
    'VolAsk5_sum', 'VolBid5_sum', 'VolImb_sum', 'VolImb_stddev',
    'VolAsk1_All', 'VolBid1_All',
    'VolRowCount_mean', 'VolRowCount_stddev',
    'VolAsk1_2345_mean', 'VolAsk1_2345_stddev',
    'VolBid1_2345_mean', 'VolBid1_2345_stddev'
]
raw_feature_dfs = {f: shape_base.copy(deep=True) for f in features}

batch_size = 500
ticker_batches = [all_tickers[i:i + batch_size] for i in range(0, len(all_tickers), batch_size)]

for date in date_list:
    date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
    print(f"\n⏳ Processing date: {date_str}")

    for batch in ticker_batches:
        tickers = [tic[:-3] for tic in batch]

        try:
            df = get_market_data(
                "XNAS", date_str, "l2",
                ticker=tickers,
                columns=[
                    "ListingId", "Ticker",
                    "BidQuantity1", "BidQuantity2", "BidQuantity3", "BidQuantity4", "BidQuantity5",
                    "AskQuantity1", "AskQuantity2", "AskQuantity3", "AskQuantity4", "AskQuantity5",
                    "EventTimestamp"
                ],
                df_engine="polars"
            )
        except Exception as e:
            print(f"❌ Failed to get data for batch on {date_str}: {e}")
            continue

        df = df.with_columns([
            sum([pl.col(c) for c in df.columns if "AskQuantity" in c]).alias("VolAsk5"),
            sum([pl.col(c) for c in df.columns if "BidQuantity" in c]).alias("VolBid5"),
        ])
        df = df.with_columns([
            ((pl.col("VolBid5") - pl.col("VolAsk5")) / (pl.col("VolBid5") + pl.col("VolAsk5"))).alias("VolImb")
        ])

        VolAsk5_sum = df["VolAsk5"].sum()
        VolBid5_sum = df["VolBid5"].sum()
        VolImb_sum = df["VolImb"].sum()
        VolImb_stddev = df["VolImb"].std()

        VolAsk1_All = df.filter(
            (pl.col("AskQuantity1") > 0) &
            (pl.col("AskQuantity2") + pl.col("AskQuantity3") + pl.col("AskQuantity4") + pl.col("AskQuantity5") == 0)
        ).height / df.height

        VolBid1_All = df.filter(
            (pl.col("BidQuantity1") > 0) &
            (pl.col("BidQuantity2") + pl.col("BidQuantity3") + pl.col("BidQuantity4") + pl.col("BidQuantity5") == 0)
        ).height / df.height

        df = df.with_columns([
            pl.col("EventTimestamp").dt.truncate("1m").alias("Minute")
        ])

        grouped = df.groupby("Minute").agg([
            pl.count().alias("RowCount"),
            ((pl.col("AskQuantity1") / (pl.col("AskQuantity2") + pl.col("AskQuantity3") + pl.col("AskQuantity4") + pl.col("AskQuantity5")).replace(0, np.nan)).fill_nan(100)).alias("VolAsk1_2345"),
            ((pl.col("BidQuantity1") / (pl.col("BidQuantity2") + pl.col("BidQuantity3") + pl.col("BidQuantity4") + pl.col("BidQuantity5")).replace(0, np.nan)).fill_nan(100)).alias("VolBid1_2345"),
        ])

        VolRowCount_mean = grouped["RowCount"].mean()
        VolRowCount_stddev = grouped["RowCount"].std()
        VolAsk1_2345_mean = grouped["VolAsk1_2345"].mean()
        VolAsk1_2345_stddev = grouped["VolAsk1_2345"].std()
        VolBid1_2345_mean = grouped["VolBid1_2345"].mean()
        VolBid1_2345_stddev = grouped["VolBid1_2345"].std()

        for f, val in zip(features, [
            VolAsk5_sum, VolBid5_sum, VolImb_sum, VolImb_stddev,
            VolAsk1_All, VolBid1_All,
            VolRowCount_mean, VolRowCount_stddev,
            VolAsk1_2345_mean, VolAsk1_2345_stddev,
            VolBid1_2345_mean, VolBid1_2345_stddev
        ]):
            raw_feature_dfs[f].loc[batch, date] = val

        del df

# ✅ 保存所有 features
for f, df in raw_feature_dfs.items():
    df.to_pickle(f"./raw_feature_{f}.pkl")
    print(f"✅ Saved {f}")
