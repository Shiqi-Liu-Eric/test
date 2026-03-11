"""alpha_naive.py — Naive Market-Cap Weighted Alpha"""

import numpy as np
import pandas as pd
from alpha_factory.alpha_utils import build_panel, get_bulk_market_caps


def generate_signal(symbols: list, start_date: str, end_date: str,
                    verbose: bool = True, panel: dict = None) -> pd.DataFrame:
    mcap = panel.get("mkt_cap") if panel else None
    if mcap is None or mcap.empty:
        mcap = get_bulk_market_caps(symbols, start_date, end_date, verbose=verbose)
    data = mcap if (mcap is not None and not mcap.empty) else (panel or build_panel(symbols, start_date, end_date, fields=["close"], verbose=verbose))["close"]
    data = data.clip(lower=0)
    print(data.div(data.sum(axis=1).replace(0, np.nan), axis=0).fillna(0))
    return data.div(data.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)