"""
alpha_wq003.py — WorldQuant Alpha #3
======================================
Formula:
    -1 * correlation(rank(open), rank(volume), 10)

Description:
    Negative 10-day rolling correlation between the cross-sectional rank
    of opening price and the cross-sectional rank of volume.

    When expensive stocks see disproportionately high volume (positive
    correlation), this alpha goes short — and vice versa.

Typical Horizon: 5–15 days
"""

import pandas as pd
from alpha_factory.alpha_utils import (
    build_panel,
    rank,
    correlation,
)


def generate_signal(
    symbols: list,
    start_date: str,
    end_date: str,
    verbose: bool = True,
    panel: dict = None,
) -> pd.DataFrame:
    """
    Compute WorldQuant Alpha #3.

    Parameters
    ----------
    symbols : list of str — ticker universe
    start_date, end_date : str 'YYYY-MM-DD'
    panel : dict, optional — pre-fetched data panel from run.py

    Returns
    -------
    pd.DataFrame : alpha signal (date × ticker)
    """
    if panel is not None and all(k in panel for k in ["open", "volume"]):
        if verbose:
            print("[Alpha WQ003] Using pre-fetched panel data...")
    else:
        if verbose:
            print("[Alpha WQ003] Fetching data from FMP...")
        panel = build_panel(
            symbols, start_date, end_date,
            fields=["open", "volume"],
            verbose=verbose,
        )

    if not panel:
        raise ValueError("No data returned from FMP API.")

    open_ = panel["open"]
    volume = panel["volume"]

    if verbose:
        print(f"[Alpha WQ003] Computing signal on {open_.shape[1]} stocks, "
              f"{open_.shape[0]} trading days...")

    # rank(open), rank(volume)
    rank_open = rank(open_)
    rank_volume = rank(volume)

    # -1 * correlation(rank(open), rank(volume), 10)
    corr_10 = correlation(rank_open, rank_volume, 10)
    alpha = -1.0 * corr_10

    if verbose:
        non_null = alpha.dropna(how="all").shape[0]
        print(f"[Alpha WQ003] Signal generated. {non_null} valid days.")

    return alpha


# Allow standalone testing
if __name__ == "__main__":
    from alpha_factory.alpha_utils import get_sp500_constituents

    symbols = get_sp500_constituents()[:30]
    signal = generate_signal(symbols, "2024-01-01", "2024-12-31")
    print(signal.tail())
    print(f"\nShape: {signal.shape}")
