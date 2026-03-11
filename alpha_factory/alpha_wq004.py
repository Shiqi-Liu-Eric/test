"""
alpha_wq004.py — WorldQuant Alpha #4
======================================
Formula:
    -1 * Ts_Rank(rank(low), 9)

Description:
    Take the cross-sectional rank of the daily low price, then compute
    its time-series rank over 9 days, and negate.

    Stocks whose low price rank has been rising (high ts_rank) are
    shorted — a short-term mean-reversion signal on the low price level.

Typical Horizon: 3–10 days
"""

import pandas as pd
from alpha_factory.alpha_utils import (
    build_panel,
    rank,
    ts_rank,
)


def generate_signal(
    symbols: list,
    start_date: str,
    end_date: str,
    verbose: bool = True,
    panel: dict = None,
) -> pd.DataFrame:
    """
    Compute WorldQuant Alpha #4.

    Parameters
    ----------
    symbols : list of str — ticker universe
    start_date, end_date : str 'YYYY-MM-DD'
    panel : dict, optional — pre-fetched data panel from run.py

    Returns
    -------
    pd.DataFrame : alpha signal (date × ticker)
    """
    if panel is not None and "low" in panel:
        if verbose:
            print("[Alpha WQ004] Using pre-fetched panel data...")
    else:
        if verbose:
            print("[Alpha WQ004] Fetching data from FMP...")
        panel = build_panel(
            symbols, start_date, end_date,
            fields=["low"],
            verbose=verbose,
        )

    if not panel:
        raise ValueError("No data returned from FMP API.")

    low = panel["low"]

    if verbose:
        print(f"[Alpha WQ004] Computing signal on {low.shape[1]} stocks, "
              f"{low.shape[0]} trading days...")

    # rank(low) — cross-sectional rank each day
    rank_low = rank(low)

    # Ts_Rank(rank(low), 9) — time-series rank over 9 days
    ts_rank_9 = ts_rank(rank_low, 9)

    # -1 * Ts_Rank(rank(low), 9)
    alpha = -1.0 * ts_rank_9

    if verbose:
        non_null = alpha.dropna(how="all").shape[0]
        print(f"[Alpha WQ004] Signal generated. {non_null} valid days.")

    return -alpha
