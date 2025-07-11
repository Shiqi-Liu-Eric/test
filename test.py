import numpy as np
import pandas as pd

def get_all_alpha_features(T, tickers, dfs):
    T_idx = dfs['weight_ftse_globalall_cap'].columns.get_loc(T)
    date_range = dfs['weight_ftse_globalall_cap'].columns[T_idx-50:T_idx]
    alphas = {f'alpha{str(i).zfill(2)}': [] for i in range(1, 16)}

    for t in date_range:
        t_idx = dfs['weight_ftse_globalall_cap'].columns.get_loc(t)
        if t_idx < 60:  # 防止前面数据不够
            for k in alphas:
                alphas[k].append(pd.Series(np.nan, index=tickers))
            continue

        # 日期切片
        close = dfs['close'].loc[tickers]
        open_ = dfs['open'].loc[tickers]
        high = dfs['high'].loc[tickers]
        low = dfs['low'].loc[tickers]
        volume = dfs['volume'].loc[tickers]
        vwap = dfs['vwap'].loc[tickers]

        close_t = close[t]
        open_t = open_[t]
        high_t = high[t]
        low_t = low[t]
        volume_t = volume[t]
        vwap_t = vwap[t]

        # Slice过去窗口
        hist_5 = dfs['close'].columns[t_idx-5:t_idx]
        hist_10 = dfs['close'].columns[t_idx-10:t_idx]
        hist_20 = dfs['close'].columns[t_idx-20:t_idx]
        hist_30 = dfs['close'].columns[t_idx-30:t_idx]
        hist_60 = dfs['close'].columns[t_idx-60:t_idx]

        # alpha01: momentum 5日
        alpha01 = dfs['close'].loc[tickers, t] / dfs['close'].loc[tickers, dfs['close'].columns[t_idx-5]] - 1

        # alpha02: slope of log(close) over 5 days
        log_close = np.log(dfs['close'].loc[tickers, hist_5])
        x = np.arange(5)
        slope = ((log_close * x).sum(axis=1) - log_close.sum(axis=1) * x.mean()) / ((x ** 2).sum() - 5 * x.mean() ** 2)
        alpha02 = slope

        # alpha03: intraday momentum normalized by range
        alpha03 = (close_t - open_t) / (high_t - low_t + 1e-6)

        # alpha04: z-score of return
        ret = np.log(dfs['close'].loc[tickers, hist_5]).diff(axis=1).iloc[:, 1:]
        alpha04 = -((ret.iloc[:, -1] - ret.mean(axis=1)) / (ret.std(axis=1) + 1e-6))

        # alpha05: close-vwap
        alpha05 = (close_t - vwap_t) / (vwap_t + 1e-6)

        # alpha06: close position in rolling high-low
        low_30 = dfs['low'].loc[tickers, hist_30].min(axis=1)
        high_30 = dfs['high'].loc[tickers, hist_30].max(axis=1)
        alpha06 = (close_t - low_30) / (high_30 - low_30 + 1e-6)

        # alpha07: 5日收益波动率
        alpha07 = ret.std(axis=1)

        # alpha08: 当天高低波幅
        alpha08 = (high_t - low_t) / (vwap_t + 1e-6)

        # alpha09: ATR(14)近似计算
        tr = np.maximum(high_t - low_t,
                        np.maximum(np.abs(high_t - close[dfs['close'].columns[t_idx-1]]),
                                   np.abs(low_t - close[dfs['close'].columns[t_idx-1]])))
        alpha09 = tr / close_t

        # alpha10: volume-close correlation (10日)
        corr_df = pd.DataFrame({
            'ret': dfs['close'].loc[tickers, hist_10].diff(axis=1).iloc[:, 1:].mean(axis=1),
            'vol': dfs['volume'].loc[tickers, hist_10].mean(axis=1)
        })
        alpha10 = corr_df['ret'].rolling(window=3).corr(corr_df['vol'])

        # alpha11: volume zscore
        vol_hist = dfs['volume'].loc[tickers, hist_10]
        alpha11 = (volume_t - vol_hist.mean(axis=1)) / (vol_hist.std(axis=1) + 1e-6)

        # alpha12: buy pressure indicator
        alpha12 = ((close_t - low_t) - (high_t - close_t)) / (high_t - low_t + 1e-6) * volume_t

        # alpha13: sharpe ratio (past 5 day)
        returns = np.log(dfs['close'].loc[tickers, hist_5]).diff(axis=1).iloc[:, 1:]
        alpha13 = returns.sum(axis=1) / (returns.std(axis=1) + 1e-6)

        # alpha14: percentage of up days in past 5
        up_days = (dfs['close'].loc[tickers, hist_5] > dfs['open'].loc[tickers, hist_5]).sum(axis=1)
        alpha14 = up_days / 5

        # alpha15: vwap momentum × volume zscore
        vwap_ret = (vwap_t - dfs['close'].loc[tickers, dfs['close'].columns[t_idx-5]]) / dfs['close'].loc[tickers, dfs['close'].columns[t_idx-5]]
        alpha15 = vwap_ret * -alpha11

        # 存入结果
        alpha_series = [alpha01, alpha02, alpha03, alpha04, alpha05, alpha06, alpha07, alpha08,
                        alpha09, alpha10, alpha11, alpha12, alpha13, alpha14, alpha15]
        for i, a in enumerate(alpha_series):
            alphas[f'alpha{str(i+1).zfill(2)}'].append(a)

    # 拼接为 MultiIndex DataFrame
    panels = []
    for k in alphas:
        df_k = pd.concat(alphas[k], axis=1)
        df_k.columns = pd.MultiIndex.from_product([[k], date_range])
        panels.append(df_k)

    result = pd.concat(panels, axis=1)
    return result
