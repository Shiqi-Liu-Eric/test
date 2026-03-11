"""alpha_funda_01.py — Fundamental Alpha: ROE / PE^2
Signal: rank(ROE / PE^2) cross-sectionally
Data source: FMP key-metrics (ROE) + ratios (PE), latest quarter
Higher ROE + lower PE => stronger long signal (value-quality composite)
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
from alpha_factory.alpha_utils import (
    build_panel, rank, fmp_get, get_historical_prices,
)


def _fetch_fundamental_snapshot(symbols, verbose=True):
    """Fetch latest quarterly ROE and PE for each symbol."""
    records = {}
    iterator = tqdm(symbols, desc="Fetching ROE/PE", disable=not verbose)
    for sym in iterator:
        try:
            km = fmp_get("key-metrics", {"symbol": sym, "period": "quarter", "limit": 1})
            rt = fmp_get("ratios", {"symbol": sym, "period": "quarter", "limit": 1})
            roe = km[0].get("returnOnEquity") if km else None
            pe = rt[0].get("priceToEarningsRatio") if rt else None
            if roe is not None and pe is not None and pe != 0:
                records[sym] = {"roe": roe, "pe": pe, "signal": roe / (pe ** 2)}
        except Exception:
            pass
    return records


def generate_signal(symbols, start_date, end_date, verbose=True, panel=None):
    if panel and "close" in panel:
        if verbose:
            print("[Alpha FUNDA01] Using pre-fetched panel data...")
    else:
        if verbose:
            print("[Alpha FUNDA01] Fetching price data from FMP...")
        panel = build_panel(symbols, start_date, end_date, fields=["close"], verbose=verbose)

    close = panel["close"]
    dates = close.index
    tickers = close.columns

    if verbose:
        print(f"[Alpha FUNDA01] Fetching fundamental data (ROE, PE) for {len(tickers)} stocks...")
    snapshot = _fetch_fundamental_snapshot(list(tickers), verbose=verbose)

    raw_signal = pd.Series({sym: snapshot[sym]["signal"] for sym in snapshot}, dtype=float)
    raw_signal = raw_signal.reindex(tickers)

    if verbose:
        valid = raw_signal.dropna()
        print(f"[Alpha FUNDA01] Valid fundamental signals: {len(valid)}/{len(tickers)}")
        if len(valid) > 0:
            print(f"  signal range: [{valid.min():.6f}, {valid.max():.6f}]")

    signal_df = pd.DataFrame(
        np.tile(raw_signal.values, (len(dates), 1)),
        index=dates, columns=tickers,
    )

    signal_df = rank(signal_df)

    if verbose:
        print(f"[Alpha FUNDA01] Signal generated. {signal_df.dropna(how='all').shape[0]} valid days.")
    return signal_df
