# ------------------------------------------------------------
# 0. 环境与输入：df_id_map, ret1, date_list 已预加载
# ------------------------------------------------------------
import pandas as pd
import numpy as np
from pathlib import Path
from pandas.tseries.offsets import BDay

from bmll2 import get_market_data_range           # Spark DF
import pyspark.sql.functions as F

OUT_DIR = Path("features")        # 保存目录
OUT_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------
# 1. 选出美国股票行，并创建所有空壳 DF
# ------------------------------------------------------------
# 1.1 过滤 index 末两位为 UN / UW
mask_us = df_id_map.index.str.endswith(("UN", "UW"))
tickers_us = df_id_map.index[mask_us]

# 1.2 基准形状 (行 = 8000+ ticker, 列 = ret1 的所有交易日)
shape_base = ret1.copy(deep=False) * np.nan        # 仅占位，不复制数据
shape_base = shape_base.loc[tickers_us]            # 仅保留美股行

# 用字典装所有 DF
raw_dfs = {}
sig_dfs = {}

# ------------------------------------------------------------
# 2. 批量抓 L2 深度 → Raw Features
# ------------------------------------------------------------
def fetch_l2_depth(mics, start, end):
    """一次性抓多 market、多日 L2，并在 Spark 内部聚合为每日每股票顶档数量和不平衡"""
    spark_df = get_market_data_range(
        markets=mics,
        start_date=start.strftime('%Y-%m-%d'),
        end_date=end.strftime('%Y-%m-%d'),
        table="l2",
        columns=(["TradeDate", "Ticker"] +
                 [f"BidQuantity{i}" for i in range(1, 6)] +
                 [f"AskQuantity{i}" for i in range(1, 6)])
    )

    # Spark 侧计算五档合计
    bid_sum = sum(F.col(f"BidQuantity{i}") for i in range(1, 6))
    ask_sum = sum(F.col(f"AskQuantity{i}") for i in range(1, 6))

    res = (spark_df
           .withColumn("VolBid5", bid_sum)
           .withColumn("VolAsk5", ask_sum)
           .withColumn("VolImb", (F.col("VolBid5") - F.col("VolAsk5")) /
                                 (F.col("VolBid5") + F.col("VolAsk5")))
           .groupBy("TradeDate", "Ticker")                 # 日聚合
           .agg(
               F.avg("VolBid5").alias("VolBid5"),
               F.avg("VolAsk5").alias("VolAsk5"),
               F.avg("VolImb").alias("VolImb")
           ))

    return res.toPandas()      # 行数≈|ticker|×|days|，已足够小可转 Pandas


# 2.1 定义抓取时间范围：ret1.columns 最小/最大
start_dt, end_dt = ret1.columns.min(), ret1.columns.max()
depth_df = fetch_l2_depth(["XNAS", "XNYS"], start_dt, end_dt)

# 2.2 转成透视表，并写入 raw_dfs 空壳
for col in ["VolBid5", "VolAsk5", "VolImb"]:
    tmp = depth_df.pivot(index="Ticker", columns="TradeDate", values=col)
    # 对齐形状并写入
    df_feat = shape_base.copy()
    df_feat.update(tmp)
    raw_dfs[col] = df_feat
    df_feat.to_pickle(OUT_DIR / f"raw_{col}.pkl")

# 👉 如需额外 Raw Feature，仿照上面在 Spark 里新增列计算即可
#    例如 Spread_Depth、HiddenRatio 等，然后再 pivot/update

# ------------------------------------------------------------
# 3. 计算 Signal Features（仅 T-5~T-1）
# ------------------------------------------------------------
# 3.1 预先生成每个 rebalance window 的“待计算日期”集合
windows = {eff: pd.bdate_range(eff - 5*BDay(), eff - BDay()) for eff in date_list}
all_need_dates = sorted(set.union(*map(set, windows.values())))

# 3.2 从 raw feature 直接衍生信号 —— 以 VolImb → Q_Imb_Top5 为例
#     其他信号（Kyle_Lambda, MOC_Imb_RelADV, Spread_Depth...）
#     可在 2. 的 Spark 中算完后再做 rolling / cross-sec 等聚合
def zscore_cross_sec(df_day):
    """横截面去极值(z)的小工具"""
    s = df_day.clip(lower=df_day.quantile(0.01), upper=df_day.quantile(0.99))
    return (s - s.mean()) / s.std(ddof=0)

sig_QImb = shape_base.copy()

for dt in all_need_dates:
    vec = raw_dfs["VolImb"][dt]             # 系列，index=ticker
    sig_QImb[dt] = zscore_cross_sec(vec)    # 横截面 zscore

sig_dfs["QImb_Top5"] = sig_QImb
sig_QImb.to_pickle(OUT_DIR / "sig_QImb_Top5.pkl")

# 👉 若要额外因子：Kyle_Lambda, Auct_Dislc...
#    1. 在 Raw 阶段多算对应原始字段或分钟条
#    2. 在此处写 rolling / 斜率 / 交互项逻辑
#    3. 同样 update 进对应 sig_df，pickle 即可

# ------------------------------------------------------------
# 4. （可选）一次性保存所有 DF
# ------------------------------------------------------------
for name, df in raw_dfs.items():
    df.to_pickle(OUT_DIR / f"raw_{name}.pkl")

for name, df in sig_dfs.items():
    df.to_pickle(OUT_DIR / f"sig_{name}.pkl")

print(f"✅ 已生成 {len(raw_dfs)} 个 Raw & {len(sig_dfs)} 个 Signal feature，保存在 {OUT_DIR.resolve()}")
