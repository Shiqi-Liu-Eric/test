"""
alpha_wq001.py — WorldQuant Alpha #1
======================================
Formula:
    rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2), 5)) - 0.5

Description:
    Conditional signal — when returns are negative, use rolling volatility;
    otherwise use price level. Then take signed power, find argmax over 5 days,
    rank cross-sectionally, and center around zero.

Typical Horizon: 1–5 days (short-term mean-reversion / vol signal)
"""

import pandas as pd
from alpha_factory.alpha_utils import (
    build_panel,
    returns as calc_returns,
    rank,
    stddev,
    signed_power,
    ts_argmax,
)


def generate_signal(
    symbols: list,
    start_date: str,
    end_date: str,
    verbose: bool = True,
    panel: dict = None,
) -> pd.DataFrame:
    """
    Compute WorldQuant Alpha #1.

    Parameters
    ----------
    symbols : list of str — ticker universe
    start_date, end_date : str 'YYYY-MM-DD'
    panel : dict, optional — pre-fetched data panel from run.py

    Returns
    -------
    pd.DataFrame : alpha signal (date × ticker), higher = stronger long signal
    """
    if panel is not None and "close" in panel:
        if verbose:
            print("[Alpha WQ001] Using pre-fetched panel data...")
    else:
        if verbose:
            print("[Alpha WQ001] Fetching data from FMP...")
        panel = build_panel(
            symbols, start_date, end_date,
            fields=["close", "volume"],
            verbose=verbose,
        )

    if not panel:
        raise ValueError("No data returned from FMP API.")

    close = panel["close"]
    ret = calc_returns(close)

    if verbose:
        print(f"[Alpha WQ001] Computing signal on {close.shape[1]} stocks, "
              f"{close.shape[0]} trading days...")

    # stddev(returns, 20)
    vol_20 = stddev(ret, 20)

    # Conditional: where returns < 0 → vol_20; else → close
    condition = ret < 0
    inner = vol_20.where(condition, close)

    # SignedPower(inner, 2)
    sp = signed_power(inner, 2.0)

    # Ts_ArgMax(sp, 5)
    argmax_5 = ts_argmax(sp, 5)

    # rank(...) - 0.5   →  centered around 0
    alpha = rank(argmax_5) - 0.5

    if verbose:
        non_null = alpha.dropna(how="all").shape[0]
        print(f"[Alpha WQ001] Signal generated. {non_null} valid days.")

    return -alpha


if __name__ == "__main__":
    from alpha_factory.alpha_utils import get_sp500_constituents
    symbols = get_sp500_constituents()[:30]  # small test universe
    print(symbols)
    signal = generate_signal(symbols, "2024-01-01", "2024-12-31")
    print(signal.tail())
    print(f"\nShape: {signal.shape}")
