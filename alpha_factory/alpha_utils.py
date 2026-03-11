"""alpha_utils.py — Shared Utilities for Alpha Factory"""
import os, yaml, requests
import numpy as np
import pandas as pd
from typing import List
from tqdm import tqdm

def load_config() -> dict:
    p = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data_preparation", "data_apiKey", "config.yaml"))
    with open(p) as f: return yaml.safe_load(f)

def get_fmp_key() -> str:
    return load_config()["keys"]["fmp_api_key"]

FMP_BASE = "https://financialmodelingprep.com/stable"

def fmp_get(endpoint: str, params: dict = None) -> list:
    params = params or {}
    params["apikey"] = get_fmp_key()
    resp = requests.get(f"{FMP_BASE}/{endpoint}", params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["historical"] if isinstance(data, dict) and "historical" in data else data

# Direct FMP constituent endpoints
_DIRECT_INDEX_EP = {
    "sp500": "sp500-constituent",
    "nasdaq": "nasdaq-constituent",
    "dowjones": "dowjones-constituent",
}

# Indices backed by ETF holdings (FMP etf/holdings endpoint)
# Maps user-friendly index name → ETF ticker that tracks it
_ETF_INDEX_MAP = {
    # ── US Broad ──
    "russell1000": "IWB",    # iShares Russell 1000
    "russell2000": "IWM",    # iShares Russell 2000
    "russell3000": "IWV",    # iShares Russell 3000
    "crsp_total":  "VTI",    # Vanguard Total Stock Market (CRSP US Total Market)
    "midcap400":   "MDY",    # SPDR S&P MidCap 400
    "smallcap600": "SLY",    # SPDR S&P SmallCap 600
    # ── Global / Intl ──
    "msci_acwi":   "ACWI",   # iShares MSCI ACWI (All Country World)
    "msci_eafe":   "EFA",    # iShares MSCI EAFE (Developed ex-US)
    "msci_em":     "EEM",    # iShares MSCI Emerging Markets
    "msci_world":  "URTH",   # iShares MSCI World
    "ftse_dev_ex_us": "VEA", # Vanguard FTSE Developed ex-US
    "ftse_ex_us":  "VEU",    # Vanguard FTSE All-World ex-US
    "ftse_global":  "VT",    # Vanguard FTSE Global All Cap (Total World)
    "ftse_em":     "VWO",    # Vanguard FTSE Emerging Markets
}

def _get_etf_holdings_symbols(etf_ticker: str) -> List[str]:
    """Fetch constituent symbols from an ETF's holdings via FMP."""
    data = fmp_get("etf/holdings", {"symbol": etf_ticker})
    if not data:
        raise ValueError(f"No ETF holdings returned for {etf_ticker}")
    symbols = []
    for d in data:
        sym = d.get("symbol") or d.get("asset")
        if sym and isinstance(sym, str) and sym.isascii():
            # skip cash / non-equity lines
            if sym.startswith("$") or sym.startswith("USD"):
                continue
            symbols.append(sym)
    return sorted(set(symbols))

def get_index_constituents(index: str = "sp500") -> List[str]:
    key = index.lower().replace(" ", "_").replace("-", "_")
    # 1. Direct FMP constituent endpoint
    if key in _DIRECT_INDEX_EP:
        return sorted([d["symbol"] for d in fmp_get(_DIRECT_INDEX_EP[key])])
    # 2. ETF-holdings backed index
    if key in _ETF_INDEX_MAP:
        etf = _ETF_INDEX_MAP[key]
        print(f"  [Universe] Resolving '{index}' via ETF holdings ({etf})...")
        return _get_etf_holdings_symbols(etf)
    # 3. Treat as raw ETF ticker
    print(f"  [Universe] Unknown index '{index}', trying as ETF ticker...")
    return _get_etf_holdings_symbols(index.upper())

def _fetch_symbol(endpoint, symbol, start_date=None, end_date=None):
    params = {"symbol": symbol}
    if start_date: params["from"] = start_date
    if end_date:   params["to"] = end_date
    data = fmp_get(endpoint, params)
    if not data: return pd.DataFrame()
    df = pd.DataFrame(data); df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)

def get_historical_prices(symbol, start_date=None, end_date=None):
    return _fetch_symbol("historical-price-eod/full", symbol, start_date, end_date)

def get_historical_market_cap(symbol, start_date=None, end_date=None):
    return _fetch_symbol("historical-market-capitalization", symbol, start_date, end_date)

def get_risk_free_rate(start_date, end_date):
    df = _fetch_symbol("historical-price-eod/full", "^IRX", start_date, end_date)
    if df.empty:
        return pd.Series(dtype=float)
    df = df.drop_duplicates("date").sort_values("date").set_index("date")
    daily_rate = (1 + df["close"] / 100) ** (1 / 252) - 1
    return daily_rate

def get_bulk_market_caps(symbols, start_date=None, end_date=None, verbose=True):
    frames = {}
    for sym in tqdm(symbols, desc="Fetching mcap", disable=not verbose):
        try:
            df = get_historical_market_cap(sym, start_date, end_date)
            if not df.empty and "marketCap" in df.columns:
                frames[sym] = df.set_index("date")["marketCap"]
        except Exception: pass
    return pd.DataFrame(frames).sort_index() if frames else pd.DataFrame()

def get_bulk_prices(symbols, start_date=None, end_date=None, verbose=True):
    frames = []
    for sym in tqdm(symbols, desc="Fetching prices", disable=not verbose):
        try:
            df = get_historical_prices(sym, start_date, end_date)
            if not df.empty: df["ticker"] = sym; frames.append(df)
        except Exception: pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

# ── Available Data Fields ──
# Price-based (historical-price-eod/full): open, high, low, close, adjClose, volume, vwap, change, changePercent
# Derived: ret = close.pct_change()
# Separate API: mkt_cap (historical-market-capitalization)
PRICE_FIELDS = {"open", "high", "low", "close", "adjClose", "volume", "vwap", "change", "changePercent"}
DERIVED_FIELDS = {"ret"}
SEPARATE_FIELDS = {"mkt_cap"}
ALL_DATA_FIELDS = PRICE_FIELDS | DERIVED_FIELDS | SEPARATE_FIELDS

def build_panel(symbols, start_date, end_date, fields=None, verbose=True):
    if fields is None: fields = ["open", "high", "low", "close", "adjClose", "volume"]
    price_fields = [f for f in fields if f in PRICE_FIELDS]
    need_close_for_ret = "ret" in fields and "close" not in price_fields
    if need_close_for_ret: price_fields.append("close")
    panel = {}
    if price_fields:
        raw = get_bulk_prices(symbols, start_date, end_date, verbose=verbose)
        if not raw.empty:
            for f in price_fields:
                if f in raw.columns:
                    panel[f] = raw.pivot(index="date", columns="ticker", values=f).sort_index()
    if "ret" in fields and "close" in panel:
        panel["ret"] = panel["close"].pct_change()
        if need_close_for_ret: del panel["close"]
    if "mkt_cap" in fields:
        mcap = get_bulk_market_caps(symbols, start_date, end_date, verbose=verbose)
        if not mcap.empty: panel["mkt_cap"] = mcap
    return panel

# ── WorldQuant-style Operators ──
def rank(df): return df.rank(axis=1, pct=True)
def delta(df, period=1): return df.diff(period)
def delay(df, period=1): return df.shift(period)
def ts_sum(df, w): return df.rolling(w, min_periods=w).sum()
def ts_mean(df, w): return df.rolling(w, min_periods=w).mean()
def ts_std(df, w): return df.rolling(w, min_periods=w).std()
def ts_max(df, w): return df.rolling(w, min_periods=w).max()
def ts_min(df, w): return df.rolling(w, min_periods=w).min()
def ts_argmax(df, w): return df.rolling(w, min_periods=w).apply(lambda x: np.argmax(x), raw=True)
def ts_argmin(df, w): return df.rolling(w, min_periods=w).apply(lambda x: np.argmin(x), raw=True)
def ts_rank(df, w): return df.rolling(w, min_periods=w).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
def correlation(x, y, w): return x.rolling(w, min_periods=w).corr(y)
def covariance(x, y, w): return x.rolling(w, min_periods=w).cov(y)
def stddev(df, w): return ts_std(df, w)
def signed_power(df, exp): return df.apply(lambda x: np.sign(x) * np.abs(x) ** exp)
def log(df): return np.log(df.where(df > 0))
def returns(close): return close.pct_change()
def scale(df, target=1.0): return df.div(df.abs().sum(axis=1).replace(0, np.nan), axis=0) * target
def sign(df): return np.sign(df)
def adv(volume, w=20): return ts_mean(volume, w)
def decay_linear(df, w):
    wt = np.arange(1, w + 1, dtype=float); wt /= wt.sum()
    return df.rolling(w, min_periods=w).apply(lambda x: np.dot(x, wt), raw=True)

# ── Operator Pipeline ──
def op_pow(signal, power=3.0): return signed_power(signal, power)
def op_decay(signal, window=5): return signal.ewm(span=window, min_periods=1).mean()
def op_truncate(signal, max_val=0.05): return signal.clip(-max_val, max_val)

def op_norm(signal, use_std=True):
    if not use_std: return rank(signal)
    mu = signal.mean(axis=1); sigma = signal.std(axis=1).replace(0, np.nan)
    return signal.sub(mu, axis=0).div(sigma, axis=0)

def op_group_neut(signal, group_map=None):
    if group_map is None: return signal.sub(signal.mean(axis=1), axis=0)
    result = signal.copy()
    for dt in signal.index:
        valid = signal.loc[dt].dropna()
        groups = group_map.reindex(valid.index).dropna()
        common = valid.index.intersection(groups.index)
        if len(common) == 0: continue
        for grp in groups[common].unique():
            t = common[groups[common] == grp]
            result.loc[dt, t] = valid[t] - valid[t].mean()
    return result

def op_risk_neut(signal, factor_dfs, panel=None):
    from numpy.linalg import lstsq
    result = signal.copy()
    for dt in signal.index:
        y = signal.loc[dt].dropna()
        if len(y) < 5: continue
        X_cols = [fdf.loc[dt].reindex(y.index) for fdf in factor_dfs.values() if dt in fdf.index]
        if not X_cols: continue
        X = pd.concat(X_cols, axis=1).dropna()
        common = y.index.intersection(X.index)
        if len(common) < 5: continue
        X_mat = np.column_stack([np.ones(len(common)), X.loc[common].values])
        try:
            beta, _, _, _ = lstsq(X_mat, y[common].values, rcond=None)
            result.loc[dt, common] = y[common].values - X_mat @ beta
        except Exception: pass
    return result

def apply_op_pipeline(signal, ops, panel=None, sector_map=None):
    for op in ops:
        t = op.get("type", "").lower()
        if t in ("oppow", "pow", "power"):
            signal = op_pow(signal, float(op.get("power", 3.0)))
        elif t in ("opgroupneut", "groupneut", "group_neut"):
            signal = op_group_neut(signal, sector_map)
        elif t in ("opneutralize", "opriskneut", "neutralize", "risk_neut"):
            pass  # return-level neutralization handled in run.py
        elif t in ("opnorm", "norm", "normalize"):
            signal = op_norm(signal, use_std=op.get("use_std", True))
        elif t in ("opdecay", "opemadecay", "decay", "ema_decay"):
            signal = op_decay(signal, int(op.get("window", 5)))
        elif t in ("optruncate", "truncate"):
            signal = op_truncate(signal, float(op.get("max_val", 0.05)))
    return signal

# ── Backtest Helpers ──
def compute_returns_from_signal(signal, forward_returns, long_short=True):
    weights = signal.copy()
    if not long_short:
        weights = weights.clip(lower=0)
    weights = weights.div(weights.abs().sum(axis=1).replace(0, np.nan), axis=0)
    pnl = (weights * forward_returns).sum(axis=1)
    return pnl, weights

def compute_stats(pnl, risk_free_daily=None, benchmark_ret=None):
    pnl = pnl.dropna()
    if len(pnl) < 2: return {}
    total_ret = (1 + pnl).prod() - 1
    ann_ret = (1 + total_ret) ** (252 / len(pnl)) - 1
    ann_vol = pnl.std() * np.sqrt(252)
    if risk_free_daily is not None and len(risk_free_daily) > 0:
        rf_aligned = risk_free_daily.reindex(pnl.index).ffill().bfill().fillna(0)
        excess_rf = pnl - rf_aligned
        sharpe = excess_rf.mean() / excess_rf.std() * np.sqrt(252) if excess_rf.std() != 0 else 0
    else:
        sharpe = pnl.mean() / pnl.std() * np.sqrt(252) if pnl.std() != 0 else 0
    if benchmark_ret is not None:
        common = pnl.index.intersection(benchmark_ret.index)
        if len(common) > 1:
            excess_bm = pnl.loc[common] - benchmark_ret.loc[common]
            ir = excess_bm.mean() / excess_bm.std() * np.sqrt(252) if excess_bm.std() != 0 else 0
        else:
            ir = 0.0
    else:
        ir = 0.0
    cum = (1 + pnl).cumprod()
    max_dd = ((cum - cum.cummax()) / cum.cummax()).min()
    downside = pnl[pnl < 0].std() * np.sqrt(252)
    sortino = ann_ret / downside if downside != 0 else 0
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
    return {
        "total_return": round(total_ret * 100, 2),
        "annual_return": round(ann_ret * 100, 2),
        "annual_volatility": round(ann_vol * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "information_ratio": round(ir, 3),
        "sortino_ratio": round(sortino, 3),
        "calmar_ratio": round(calmar, 3),
        "max_drawdown": round(max_dd * 100, 2),
        "num_trading_days": len(pnl),
    }
