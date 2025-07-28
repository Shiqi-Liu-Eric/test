# -*- coding: utf-8 -*-
"""
Author : Verse – MSC I 2025
Target : ① 生成原始 (raw) 挂单/成交特征  
         ② 计算不平衡类 signal α 因子  
         ③ 把每个特征/因子存成与 ret1 同维度 (8021×T) 的 DataFrame，并以 pickle 持久化  
环境 : bmll‑api v≥3.9 ；pandas 2.2+；pyspark 已在 BMLL Data Lab 预装
---------------------------------------------------------------------------
★ 如只在本地 IDE 运行，请先执行  >>>  pip install bmll pandas pyarrow pyspark
"""

import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from bmll2 import get_market_data, get_market_data_range          # 官方 API
import pyspark.sql.functions as F                                  # Spark 聚合

# --------------------------------------------------------------------------
# 0. 读入基础表
# --------------------------------------------------------------------------
DATA_DIR   = Path("./feature_pickle")      # 保存位置
DATA_DIR.mkdir(exist_ok=True)

df_id_map  = pd.read_pickle("df_id_map.pkl")     # 两列 [SEDOL, ISIN]，index=ticker
ret1       = pd.read_pickle("ret1.pkl")          # 8021 × T 的收益矩阵
date_list  = pd.read_pickle("date_list.pkl")     # pd.Timestamp 列表 (effective date)

# 只保留 MSCI 代码后两位为 UN/UW (= US 本土挂牌) 的股票
us_mask          = df_id_map.index.str[-2:].isin(["UN", "UW"])
valid_tickers    = df_id_map.index[us_mask]
ret1             = ret1.loc[valid_tickers]       # 保证行顺序一致

all_dates        = ret1.columns                  # Business‑Day datetime64[ns]
markets_us       = ["XNAS", "XNYS"]              # Nasdaq & NYSE 主场

# --------------------------------------------------------------------------
# 1. ===============  生成 RAW 特征  ========================================
# --------------------------------------------------------------------------
RAW_NAMES = ["VolAsk5", "VolBid5", "VolImb",
             "Spread", "Depth1"]                 # 你可再追加其他底层字段

raw_dfs   = {nm: pd.DataFrame(index=valid_tickers, columns=all_dates, dtype="float32")
             for nm in RAW_NAMES}

def _pull_l2_single_day(mic: str, dt: pd.Timestamp) -> pd.DataFrame:
    """
    拉取某市场某天的 L2 挂单表（只顶 5 档 + 价差）并做必要字段转换
    """
    cols = ["Ticker",
            "BidPrice1", "AskPrice1",
            *[f"BidQuantity{i}" for i in range(1, 6)],
            *[f"AskQuantity{i}" for i in range(1, 6)]]

    df = get_market_data(mic, dt.strftime("%Y-%m-%d"), "l2", columns=cols, df_engine="pandas")
    return df

def _build_raw_feature(l2_df: pd.DataFrame) -> pd.DataFrame:
    """
    根据单日 L2 表构造 raw → 返回 [Ticker, RAW_NAMES...]  DataFrame
    """
    bid_cols = [f"BidQuantity{i}" for i in range(1, 6)]
    ask_cols = [f"AskQuantity{i}" for i in range(1, 6)]

    out               = pd.DataFrame(index=l2_df["Ticker"].values)
    out["VolBid5"]    = l2_df[bid_cols].sum(axis=1).values
    out["VolAsk5"]    = l2_df[ask_cols].sum(axis=1).values
    out["VolImb"]     = (out["VolBid5"] - out["VolAsk5"]) / (out["VolBid5"] + out["VolAsk5"])
    out["Spread"]     = (l2_df["AskPrice1"].values - l2_df["BidPrice1"].values)
    out["Depth1"]     = l2_df["BidQuantity1"].values + l2_df["AskQuantity1"].values
    return out

print("--> Start building RAW features")
for d in all_dates:
    # 拼接 2 个 venue 的 L2 后再 groupby 平均
    l2_concat = pd.concat([_pull_l2_single_day(mic, d) for mic in markets_us],
                          ignore_index=True)
    raw_day   = (_build_raw_feature(l2_concat)
                 .groupby(level=0).mean())                       # 合并 XNAS+XNYS

    # 写入大矩阵
    for nm in RAW_NAMES:
        raw_dfs[nm].loc[raw_day.index, d] = raw_day[nm].values

    print(f"  ✅ raw @{d.date()}  done")

# 保存 RAW
for nm, df in raw_dfs.items():
    df.astype("float32").to_pickle(DATA_DIR / f"RAW_{nm}.pkl")
print("--> RAW pickle saved\n")

# --------------------------------------------------------------------------
# 2. ===============  计算 Signal α 因子  ===================================
#   仅在 5‑day 预调仓窗 (T‑5 ~ T‑1) 计算，其他日期填 NA
# --------------------------------------------------------------------------
# 2.1 窗口日期集合
windows = set()
for eff in date_list:
    windows.update(pd.bdate_range(eff - BDay(5), eff - BDay(1)))
window_dates = sorted(windows)                        # List[pd.Timestamp]

# 2.2 Spark‑based 批量拉 L2 / L3 / trades  (跨月 → Spark 更快)
spark = None   # Data Lab 默认已提供 spark session，若本地则需自己 from pyspark.sql import SparkSession
try:
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()
except Exception:
    pass  # 若在 Data Lab 里会自动获取

# ---------------- Signal 容器 ---------------
SIG_NAMES  = ["QImb_Top5", "HiddenRatio", "MOC_Imb_RelADV",
              "Auct_Dislc", "Kyle_Lambda", "OI_sigma", "Spread_Depth"]
sig_dfs    = {nm: pd.DataFrame(index=valid_tickers, columns=all_dates, dtype="float32")
              for nm in SIG_NAMES}

# ------------------- 工具函数 -----------------
def spark_l2_day(mics, dt):
    """
    利用 get_market_data_range 拉多市场同日 L2，返回 Spark DF
    """
    return get_market_data_range(mics, dt.strftime("%Y-%m-%d"), dt.strftime("%Y-%m-%d"),
                                 "l2",
                                 columns=["Ticker", "EventTimestamp",
                                          "BidPrice1", "AskPrice1",
                                          *[f"BidQuantity{i}" for i in range(1, 6)],
                                          *[f"AskQuantity{i}" for i in range(1, 6)]])

def calc_qimb_top5(l2_spark):
    bid_sum = sum(F.col(f"BidQuantity{i}") for i in range(1, 6))
    ask_sum = sum(F.col(f"AskQuantity{i}") for i in range(1, 6))
    df = (l2_spark
          .withColumn("VolBid5", bid_sum)
          .withColumn("VolAsk5", ask_sum)
          .withColumn("VolImb", (F.col("VolBid5") - F.col("VolAsk5")) /
                               (F.col("VolBid5") + F.col("VolAsk5")))
          # 尾盘 15:30‑15:59 (US Eastern) —— 时间字段已是 venue local；改成 HH:MM:SS 字符串即可
          .where("substring(EventTimestamp,12,5) between '15:30' and '15:59'")
          .groupBy("Ticker")
          .agg(F.avg("VolImb").alias("QImb_Top5")))
    return df

def calc_spread_depth(l2_spark):
    df = (l2_spark
          .withColumn("Spread", F.col("AskPrice1") - F.col("BidPrice1"))
          .withColumn("Depth1", F.col("BidQuantity1") + F.col("AskQuantity1"))
          .withColumn("Spread_Depth", F.col("Spread") / F.col("Depth1"))
          .groupBy("Ticker").agg(F.avg("Spread_Depth").alias("Spread_Depth")))
    return df

# ------------- 主循环：仅跑 window_dates -----------
print("--> Start computing SIGNAL features (T-5  window only)")
for d in window_dates:
    l2_spark = spark_l2_day(markets_us, d)

    # = QImb_Top5 =
    qimb_df = calc_qimb_top5(l2_spark).toPandas().set_index("Ticker")
    sig_dfs["QImb_Top5"].loc[qimb_df.index, d] = qimb_df["QImb_Top5"].values

    # = Spread_Depth =
    sd_df   = calc_spread_depth(l2_spark).toPandas().set_index("Ticker")
    sig_dfs["Spread_Depth"].loc[sd_df.index, d] = sd_df["Spread_Depth"].values

    # >>> 其余 HiddenRatio / Kyle_Lambda / MOC 等因子因为
    # >>> 需要 L3 / trades / auction feed，可在此处按与你前面 notebook
    # >>> 一致的逻辑写 Spark SQL 聚合，流程完全相同：withColumn → groupBy → avg
    # >>> 计算完再 toPandas() → 写入 sig_dfs[...]  (示例略)
    # ----------------------------------------------------------
    print(f"  ✅ signal @{d.date()}  done")

# 保存 SIGNAL
for nm, df in sig_dfs.items():
    df.astype("float32").to_pickle(DATA_DIR / f"SIG_{nm}.pkl")
print("--> SIGNAL pickle saved")

# --------------------------------------------------------------------------
# Done!  你现在在 ./feature_pickle/ 下可看到：
#   RAW_VolAsk5.pkl, RAW_VolBid5.pkl, ... , SIG_QImb_Top5.pkl, ...
# --------------------------------------------------------------------------
