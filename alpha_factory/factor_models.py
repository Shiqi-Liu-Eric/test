"""factor_models.py — Risk Factor Computation & Return-Level Neutralization
Computes common risk factors from panel data, estimates daily factor premiums
via Fama-MacBeth cross-sectional regression, and neutralizes portfolio/benchmark
returns by removing the factor-driven component.
"""
import numpy as np
import pandas as pd
from numpy.linalg import lstsq


def compute_factor_exposures(panel, factor_names):
    close = panel.get("close")
    if close is None:
        return {}
    ret = close.pct_change()
    volume = panel.get("volume")
    exposures = {}

    for fn in factor_names:
        fl = fn.lower()
        if fl in ("momentum", "mom"):
            exposures[fn] = ret.rolling(20, min_periods=10).sum()
        elif fl in ("short_momentum", "mom5", "momentum5"):
            exposures[fn] = ret.rolling(5, min_periods=3).sum()
        elif fl in ("volatility", "vol"):
            exposures[fn] = ret.rolling(20, min_periods=10).std()
        elif fl in ("size", "market_cap"):
            mc = panel.get("mkt_cap")
            exposures[fn] = np.log(mc.replace(0, np.nan)) if mc is not None else np.log(close.replace(0, np.nan))
        elif fl in ("liquidity", "liq"):
            if volume is not None:
                exposures[fn] = np.log1p(volume.rolling(20, min_periods=10).mean())
        elif fl in ("reversal", "short_term_reversal", "str"):
            exposures[fn] = -ret.rolling(5, min_periods=3).sum()
        elif fl in ("turnover", "tvr"):
            if volume is not None:
                exposures[fn] = volume.rolling(5, min_periods=3).mean()
    return exposures


def compute_factor_premiums(stock_returns, factor_exposures):
    """Fama-MacBeth: cross-sectional regression each day → daily factor premium series."""
    fnames = list(factor_exposures.keys())
    if not fnames:
        return {}
    premiums = {fn: {} for fn in fnames}

    for dt in stock_returns.index:
        y = stock_returns.loc[dt].dropna()
        if len(y) < 10:
            continue
        cols, valid = [], []
        for fn in fnames:
            exp = factor_exposures[fn]
            if dt not in exp.index:
                continue
            cols.append(exp.loc[dt].reindex(y.index))
            valid.append(fn)
        if not cols:
            continue
        X = pd.concat(cols, axis=1).dropna()
        common = y.index.intersection(X.index)
        if len(common) < max(10, len(valid) + 2):
            continue
        Xv = X.loc[common].values.astype(float)
        mu, sig = Xv.mean(0), Xv.std(0)
        sig[sig == 0] = 1.0
        Xz = (Xv - mu) / sig
        Xm = np.column_stack([np.ones(len(common)), Xz])
        try:
            beta, _, _, _ = lstsq(Xm, y[common].values, rcond=None)
            for i, fn in enumerate(valid):
                premiums[fn][dt] = beta[i + 1] / sig[i]
        except Exception:
            pass

    return {fn: pd.Series(v).sort_index() for fn, v in premiums.items() if v}


def neutralize_portfolio_returns(pnl, weights, factor_exposures, factor_premiums):
    """Subtract factor-driven component from portfolio returns.
    factor_component_t = sum_k( (w_t · exposure_k_t) * premium_k_t )
    """
    fc = pd.Series(0.0, index=pnl.index)
    for fn, prem in factor_premiums.items():
        if fn not in factor_exposures:
            continue
        exp = factor_exposures[fn]
        cd = pnl.index.intersection(exp.index).intersection(weights.index).intersection(prem.index)
        ct = weights.columns.intersection(exp.columns)
        wa = weights.loc[cd, ct].fillna(0)
        ea = exp.loc[cd, ct].fillna(0)
        port_exp = (wa * ea).sum(axis=1)
        fc.loc[cd] += port_exp * prem.reindex(cd).fillna(0)
    return pnl - fc, fc


def neutralize_benchmark_returns(benchmark_ret, factor_premiums, window=60):
    """Remove factor exposure from benchmark via rolling time-series regression."""
    if benchmark_ret is None or not factor_premiums:
        return benchmark_ret
    fdf = pd.DataFrame(factor_premiums)
    common = benchmark_ret.index.intersection(fdf.index)
    if len(common) < 30:
        return benchmark_ret
    bm = benchmark_ret.reindex(common)
    F = fdf.reindex(common).fillna(0)
    result = benchmark_ret.copy()
    for i in range(window, len(common)):
        sl = common[max(0, i - window):i + 1]
        y = bm.loc[sl].values
        X = np.column_stack([np.ones(len(sl)), F.loc[sl].values])
        try:
            beta, _, _, _ = lstsq(X, y, rcond=None)
            dt = common[i]
            result.loc[dt] = bm.loc[dt] - F.loc[dt].values @ beta[1:]
        except Exception:
            pass
    return result
