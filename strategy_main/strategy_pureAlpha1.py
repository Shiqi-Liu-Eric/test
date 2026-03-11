"""strategy_pureAlpha1.py — Pure Alpha Strategy: combine all alphas from backtest_global.xml into a live weight list."""
import sys, os, argparse, importlib
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from alpha_factory.alpha_utils import (
    get_index_constituents, build_panel, fmp_get,
    rank, compute_returns_from_signal,
    apply_op_pipeline, op_norm,
)

# ── XML Parsing (reuse run.py logic) ──

def _parse_alpha_ops(alpha_el):
    ops = []
    op_el = alpha_el.find("Op")
    if op_el is None:
        return ops
    for child in op_el:
        entry = {"type": child.tag}
        entry.update(child.attrib)
        factors = [f.text.strip() for f in child.findall("factor") if f.text]
        if factors:
            entry["factors"] = factors
        ops.append(entry)
    return ops

def _parse_alpha_data(alpha_el):
    data_el = alpha_el.find("Data")
    if data_el is None:
        return ["close", "volume"]
    raw_fields = [f.text.strip() for f in data_el.findall("field") if f.text]
    fields = []
    for entry in raw_fields:
        for part in entry.split(","):
            part = part.strip()
            if part:
                fields.append(part)
    return fields if fields else ["close", "volume"]

def parse_global_config(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    def text(path, default=None, base=None):
        el = (base or root).find(path)
        return el.text.strip() if el is not None and el.text else default

    config = {
        "name": text("meta/name", "alpha_global"),
        "universe_index": text("universe/index", "sp500"),
        "max_stocks": int(text("universe/max_stocks", "0")),
        "start_date": text("dates/start", "2020-01-01"),
        "end_date": text("dates/end", "2025-12-31"),
    }

    alphas_el = root.find("alphas")
    config["alphas"] = []
    if alphas_el is not None:
        for a in alphas_el.findall("Alpha"):
            info_el = a.find("AlphaInfo")
            info = dict(info_el.attrib) if info_el is not None else {}
            config["alphas"].append({
                "module": text("module", base=a),
                "delay": int(text("delay", "1", base=a)),
                "intervalmode": text("intervalmode", "daily", base=a),
                "ops": _parse_alpha_ops(a),
                "data_fields": _parse_alpha_data(a),
                "alpha_info": info,
            })
    return config

# ── Core Strategy ──

def compute_weights(config, lookback_days=120, target_date=None, verbose=True):
    """
    Compute today's portfolio weights by:
      1. Fetching recent data (lookback_days for operator warm-up)
      2. Running each alpha's generate_signal()
      3. Applying Op pipeline
      4. Combining signals (equal-weight across alphas)
      5. Converting final signal to portfolio weights (long/short, dollar-neutral)

    Returns
    -------
    weights_df : pd.DataFrame  — single-row DataFrame (date × tickers) of portfolio weights
    detail     : dict          — per-alpha signal snapshots for audit
    """
    if target_date is None:
        target_date = datetime.today().strftime("%Y-%m-%d")
    end_dt = pd.Timestamp(target_date)
    start_dt = end_dt - timedelta(days=lookback_days + 60)
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    if verbose:
        print(f"[PureAlpha1] target_date={target_date}  lookback={lookback_days}d")
        print(f"[PureAlpha1] data window: {start_str} → {end_str}")

    # 1. Universe
    symbols = get_index_constituents(config["universe_index"])
    if config["max_stocks"] > 0:
        symbols = symbols[:config["max_stocks"]]
    if verbose:
        print(f"[PureAlpha1] Universe: {len(symbols)} stocks ({config['universe_index']})")

    # 2. Build shared data panel
    all_fields = set()
    for a in config["alphas"]:
        all_fields.update(a.get("data_fields", ["close", "volume"]))
    all_fields.add("close")
    if verbose:
        print(f"[PureAlpha1] Fetching panel ({', '.join(sorted(all_fields))})...")
    panel = build_panel(symbols, start_str, end_str, fields=sorted(all_fields), verbose=verbose)
    if "close" not in panel:
        raise ValueError("Failed to fetch close prices.")

    # 3. Run each alpha
    n_alphas = len(config["alphas"])
    all_signals = []
    detail = {}

    for idx, alpha_cfg in enumerate(config["alphas"], 1):
        module_name = alpha_cfg["module"]
        ops = alpha_cfg["ops"]
        info = alpha_cfg.get("alpha_info", {})
        alpha_name = info.get("name", f"alpha_{idx}")

        if verbose:
            print(f"  [{idx}/{n_alphas}] {alpha_name} ({module_name})")

        alpha_mod = importlib.import_module(module_name)
        signal = alpha_mod.generate_signal(
            symbols=symbols, start_date=start_str, end_date=end_str,
            verbose=False, panel=panel,
        )

        # align to common dates/tickers
        close = panel["close"]
        common_dates = signal.index.intersection(close.index)
        common_tickers = signal.columns.intersection(close.columns)
        signal = signal.loc[common_dates, common_tickers]

        # apply Op pipeline (skip OpNeutralize at strategy level — it's handled in backtest only)
        signal = apply_op_pipeline(signal, ops, panel=panel, sector_map=None)

        all_signals.append(signal)
        detail[alpha_name] = signal

    # 4. Combine — equal-weight average across alphas, then cross-sectional z-score
    if len(all_signals) == 1:
        combined = all_signals[0]
    else:
        # align all signal DataFrames
        common_idx = all_signals[0].index
        common_cols = all_signals[0].columns
        for s in all_signals[1:]:
            common_idx = common_idx.intersection(s.index)
            common_cols = common_cols.intersection(s.columns)
        stacked = np.stack([s.loc[common_idx, common_cols].values for s in all_signals], axis=0)
        combined = pd.DataFrame(
            np.nanmean(stacked, axis=0),
            index=common_idx, columns=common_cols,
        )

    # final cross-sectional z-score
    combined = op_norm(combined, use_std=True)

    # 5. Latest-day weights
    latest_date = combined.index[-1]
    latest_signal = combined.loc[latest_date].dropna()

    # dollar-neutral weights: sum(abs(w)) = 1
    abs_sum = latest_signal.abs().sum()
    if abs_sum == 0:
        weights = latest_signal * 0
    else:
        weights = latest_signal / abs_sum

    weights = weights.sort_values(ascending=False)
    weights_df = weights.to_frame(name="weight").rename_axis("ticker")
    weights_df["date"] = latest_date.strftime("%Y-%m-%d")
    weights_df = weights_df[["date", "weight"]]

    if verbose:
        n_long = (weights > 0).sum()
        n_short = (weights < 0).sum()
        gross = weights.abs().sum()
        net = weights.sum()
        print(f"\n[PureAlpha1] Weights for {latest_date.strftime('%Y-%m-%d')}:")
        print(f"  Stocks: {len(weights)} (long={n_long}, short={n_short})")
        print(f"  Gross exposure: {gross:.4f}  |  Net exposure: {net:.4f}")
        print(f"\n  Top 5 long:")
        for tk, w in weights.head(5).items():
            print(f"    {tk:>8s}  {w:>+.6f}")
        print(f"  Top 5 short:")
        for tk, w in weights.tail(5).items():
            print(f"    {tk:>8s}  {w:>+.6f}")

    return weights_df, detail

# ── Identifier Lookup ──

def fetch_identifiers(symbols, verbose=True):
    id_map = {}
    if verbose:
        print(f"[PureAlpha1] Fetching cusip/isin for {len(symbols)} tickers...")
    for sym in symbols:
        try:
            data = fmp_get("profile", {"symbol": sym})
            if data:
                d = data[0] if isinstance(data, list) else data
                id_map[sym] = {
                    "cusip": d.get("cusip", ""),
                    "isin": d.get("isin", ""),
                }
            else:
                id_map[sym] = {"cusip": "", "isin": ""}
        except Exception:
            id_map[sym] = {"cusip": "", "isin": ""}
    return id_map

# ── Save ──

def save_weights(weights_df, out_dir=None, tag="strategy_pureAlpha1", verbose=True):
    date_str = weights_df["date"].iloc[0].replace("-", "")
    if out_dir is None:
        out_dir = os.path.join(REPO_ROOT, "execution_main", "trading_log", date_str)
    os.makedirs(out_dir, exist_ok=True)

    tickers = weights_df.index.tolist()
    id_map = fetch_identifiers(tickers, verbose=verbose)

    fname = f"{tag}_weight.csv"
    fpath = os.path.join(out_dir, fname)
    with open(fpath, "w") as f:
        f.write("ticker,cusip,isin,weight,order_time,order_type\n")
        for tk in tickers:
            w = weights_df.loc[tk, "weight"]
            cusip = id_map.get(tk, {}).get("cusip", "")
            isin = id_map.get(tk, {}).get("isin", "")
            f.write(f"{tk},{cusip},{isin},{w},MOC,default\n")
    print(f"[PureAlpha1] Weights saved → {fpath} ({len(tickers)} rows)")
    return fpath

# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="Pure Alpha Strategy — Weight Generator")
    parser.add_argument("--config", "-c", default="backtest_main/backtest_global.xml",
                        help="Path to global alpha XML config")
    parser.add_argument("--date", "-d", default=None,
                        help="Target date (YYYY-MM-DD). Default: today")
    parser.add_argument("--lookback", "-l", type=int, default=120,
                        help="Lookback days for operator warm-up (default: 120)")
    parser.add_argument("--save", "-s", action="store_true", default=True,
                        help="Save weights CSV (default: True)")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not save weights CSV")
    args = parser.parse_args()

    config_path = args.config if os.path.isabs(args.config) else os.path.join(REPO_ROOT, args.config)
    if not os.path.exists(config_path):
        print(f"[ERROR] Config not found: {config_path}")
        sys.exit(1)

    config = parse_global_config(config_path)
    print(f"\n{'=' * 60}")
    print(f"  PURE ALPHA STRATEGY: {config['name']}")
    print(f"  Alphas: {len(config['alphas'])}  |  Universe: {config['universe_index']}")
    print(f"{'=' * 60}\n")

    weights_df, detail = compute_weights(
        config,
        lookback_days=args.lookback,
        target_date=args.date,
        verbose=True,
    )

    if args.save and not args.no_save:
        save_weights(weights_df)

    return weights_df

if __name__ == "__main__":
    main()
