"""
alpha_wq005.py — WorldQuant Alpha #5
======================================
Formula:
    rank(open - (sum(vwap, 10) / 10)) * (-1 * abs(rank(close - vwap)))

Description:
    Interaction between two terms:
      1. rank(open - 10-day mean VWAP)  — how far today's open is above/below
         the recent average VWAP.
      2. -abs(rank(close - vwap))       — magnitude of the close-to-vwap deviation
         (always negative, penalising large deviations).

    Product captures mean-reversion when open deviates from VWAP trend
    and penalises excessive close-to-vwap gaps.

Data: open, close, vwap
Typical Horizon: 3–10 days
"""

import pandas as pd
from alpha_factory.alpha_utils import (
    build_panel,
    rank,
    ts_mean,
)


def generate_signal(
    symbols: list,
    start_date: str,
    end_date: str,
    verbose: bool = True,
    panel: dict = None,
) -> pd.DataFrame:
    """
    Compute WorldQuant Alpha #5.

    Parameters
    ----------
    symbols : list of str — ticker universe
    start_date, end_date : str 'YYYY-MM-DD'
    panel : dict, optional — pre-fetched data panel from run.py

    Returns
    -------
    pd.DataFrame : alpha signal (date × ticker)
    """
    required = ["open", "close", "vwap"]
    if panel is not None and all(k in panel for k in required):
        if verbose:
            print("[Alpha WQ005] Using pre-fetched panel data...")
    else:
        if verbose:
            print("[Alpha WQ005] Fetching data from FMP...")
        panel = build_panel(
            symbols, start_date, end_date,
            fields=required,
            verbose=verbose,
        )

    if not panel:
        raise ValueError("No data returned from FMP API.")

    open_ = panel["open"]
    close = panel["close"]
    vwap = panel["vwap"]

    if verbose:
        print(f"[Alpha WQ005] Computing signal on {close.shape[1]} stocks, "
              f"{close.shape[0]} trading days...")

    # sum(vwap, 10) / 10  =  ts_mean(vwap, 10)
    vwap_mean_10 = ts_mean(vwap, 10)

    # rank(open - ts_mean(vwap, 10))
    term1 = rank(open_ - vwap_mean_10)

    # -1 * abs(rank(close - vwap))
    term2 = -1.0 * rank(close - vwap).abs()

    # product
    alpha = term1 * term2

    if verbose:
        non_null = alpha.dropna(how="all").shape[0]
        print(f"[Alpha WQ005] Signal generated. {non_null} valid days.")

    return -alpha
