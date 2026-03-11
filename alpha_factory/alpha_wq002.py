"""
alpha_wq002.py — WorldQuant Alpha #2
======================================
Formula:
    -1 * correlation(rank(delta(log(volume), 2)), rank((close - open) / open), 6)

Description:
    Negative rolling correlation between:
      - rank of 2-day change in log volume
      - rank of intraday return (close-to-open pct)
    over a 6-day window.

    Captures divergence between volume acceleration and intraday price movement.

Typical Horizon: 3–10 days
"""

import pandas as pd
import numpy as np
from alpha_factory.alpha_utils import (
    build_panel,
    rank,
    delta,
    correlation,
    log,
)


def generate_signal(
    symbols: list,
    start_date: str,
    end_date: str,
    verbose: bool = True,
    panel: dict = None,
) -> pd.DataFrame:
    """
    Compute WorldQuant Alpha #2.

    Parameters
    ----------
    symbols : list of str — ticker universe
    start_date, end_date : str 'YYYY-MM-DD'
    panel : dict, optional — pre-fetched data panel from run.py

    Returns
    -------
    pd.DataFrame : alpha signal (date × ticker)
    """
    if panel is not None and all(k in panel for k in ["open", "close", "volume"]):
        if verbose:
            print("[Alpha WQ002] Using pre-fetched panel data...")
    else:
        if verbose:
            print("[Alpha WQ002] Fetching data from FMP...")
        panel = build_panel(
            symbols, start_date, end_date,
            fields=["open", "close", "volume"],
            verbose=verbose,
        )

    if not panel:
        raise ValueError("No data returned from FMP API.")

    open_ = panel["open"]
    close = panel["close"]
    volume = panel["volume"]

    if verbose:
        print(f"[Alpha WQ002] Computing signal on {close.shape[1]} stocks, "
              f"{close.shape[0]} trading days...")

    # rank(delta(log(volume), 2))
    log_vol = log(volume)
    delta_log_vol = delta(log_vol, 2)
    rank_delta_log_vol = rank(delta_log_vol)

    # rank((close - open) / open)
    intraday_ret = (close - open_) / open_.replace(0, np.nan)
    rank_intraday = rank(intraday_ret)

    # -1 * correlation(..., ..., 6)
    corr_6 = correlation(rank_delta_log_vol, rank_intraday, 6)
    alpha = -1.0 * corr_6

    if verbose:
        non_null = alpha.dropna(how="all").shape[0]
        print(f"[Alpha WQ002] Signal generated. {non_null} valid days.")

    return -alpha


# Allow standalone testing
if __name__ == "__main__":
    from alpha_factory.alpha_utils import get_sp500_constituents

    symbols = get_sp500_constituents()[:30]
    signal = generate_signal(symbols, "2024-01-01", "2024-12-31")
    print(signal.tail())
    print(f"\nShape: {signal.shape}")
