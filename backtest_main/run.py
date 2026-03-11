"""run.py — Backtest Runner (Main Entry Point)"""
import sys, os, argparse, importlib
import xml.etree.ElementTree as ET
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path: sys.path.insert(0, REPO_ROOT)

from alpha_factory.alpha_utils import (
    get_index_constituents, get_historical_prices, build_panel,
    get_risk_free_rate,
    returns as calc_returns, rank, compute_returns_from_signal,
    compute_stats, apply_op_pipeline,
    PRICE_FIELDS, DERIVED_FIELDS, SEPARATE_FIELDS, ALL_DATA_FIELDS,
)
from alpha_factory.factor_models import (
    compute_factor_exposures, compute_factor_premiums,
    neutralize_portfolio_returns, neutralize_benchmark_returns,
)

# ── XML Config Parser ──

def _parse_alpha_ops(alpha_el):
    ops = []
    op_el = alpha_el.find("Op")
    if op_el is None: return ops
    for child in op_el:
        entry = {"type": child.tag}
        entry.update(child.attrib)
        factors = [f.text.strip() for f in child.findall("factor") if f.text]
        if factors: entry["factors"] = factors
        ops.append(entry)
    return ops

def _parse_alpha_data(alpha_el):
    data_el = alpha_el.find("Data")
    if data_el is None: return ["close", "volume"]
    raw_fields = [f.text.strip() for f in data_el.findall("field") if f.text]
    # Support comma-separated fields inside a single <field> tag
    fields = []
    for entry in raw_fields:
        for part in entry.split(","):
            part = part.strip()
            if part:
                fields.append(part)
    return fields if fields else ["close", "volume"]

def parse_config(xml_path):
    tree = ET.parse(xml_path); root = tree.getroot()
    def text(path, default=None, base=None):
        el = (base or root).find(path)
        return el.text.strip() if el is not None and el.text else default
    def boolean(path, default=False, base=None):
        val = text(path, base=base)
        return val.lower() in ("true", "1", "yes") if val else default

    config = {
        "name": text("meta/name", "unnamed"),
        "description": text("meta/description", ""),
        "universe_index": text("universe/index", "sp500"),
        "max_stocks": int(text("universe/max_stocks", "0")),
        "start_date": text("dates/start", "2023-01-01"),
        "end_date": text("dates/end", "2025-12-31"),
        "strategy_type": text("strategy/type", "long_short_zscore"),
        "long_short": boolean("strategy/long_short", True),
        "ir_benchmark": text("ir_benchmark/type", "risk_free"),
        "sharpe_benchmark": text("sharpe_benchmark/type", "risk_free"),
        "commission_bps": float(text("costs/commission_bps", "0")),
        "slippage_bps": float(text("costs/slippage_bps", "0")),
        "print_stats": boolean("output/print_stats", True),
        "print_summary_table": boolean("output/print_summary_table", True),
        "print_daily_detail": boolean("output/print_daily_detail", False),
        "plot_equity_curve": boolean("output/plot_equity_curve", True),
        "plot_group_returns": boolean("output/plot_group_returns", False),
        "save_signal_csv": boolean("output/save_signal_csv", False),
        "save_pnl": boolean("output/save_pnl", False),
    }
    alphas_el = root.find("alphas")
    if alphas_el is not None:
        config["alphas"] = []
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
    else:
        alpha_el = root.find("alpha")
        info_el = alpha_el.find("AlphaInfo") if alpha_el is not None else None
        info = dict(info_el.attrib) if info_el is not None else {}
        config["alphas"] = [{
            "module": text("alpha/module"),
            "delay": int(text("alpha/delay", "1")),
            "intervalmode": text("alpha/intervalmode", text("parameters/rebalance_frequency", "daily")),
            "ops": _parse_alpha_ops(alpha_el) if alpha_el else [],
            "data_fields": ["close", "volume"],
            "alpha_info": info,
        }]
        if root.find("strategy") is None:
            config["long_short"] = boolean("parameters/long_short", True)
    return config

# ── Strategy ──

def _strategy_long_short_zscore(signal, fwd_ret, long_short=True):
    return compute_returns_from_signal(signal, fwd_ret, long_short=long_short)

STRATEGIES = {"long_short_zscore": _strategy_long_short_zscore, "long_short": _strategy_long_short_zscore}

# ── Interval-Mode Resampling ──

def _apply_intervalmode(signal, mode):
    mode = mode.lower()
    if mode == "daily": return signal
    if mode == "weekly":
        rebal_mask = signal.index.to_series().dt.dayofweek == 4
        if not rebal_mask.any():
            wg = signal.index.to_series().dt.isocalendar().week
            rebal_mask = signal.index == signal.index.to_series().groupby(wg).transform("last")
    elif mode == "monthly":
        mg = signal.index.to_series().dt.to_period("M")
        rebal_mask = signal.index == signal.index.to_series().groupby(mg).transform("last")
    else:
        return signal
    resampled = signal.copy()
    resampled.loc[~rebal_mask] = np.nan
    return resampled.ffill()

# ── Benchmark ──

def _fetch_benchmark(symbol, start, end):
    prices = get_historical_prices(symbol, start, end)
    if prices is None or prices.empty: return None
    prices = prices.sort_values("date").set_index("date")
    col = "adjClose" if "adjClose" in prices.columns else "close" if "close" in prices.columns else None
    if col is None: return None
    close = prices[col]; close.index = pd.to_datetime(close.index)
    return close.pct_change().dropna()

# ── Backtest Engine ──

def run_backtest(config, no_plot=False, save=False):
    name = config["name"]
    print("=" * 60)
    print(f"  BACKTEST: {name} — {config['description']}")
    print(f"  Alphas: {len(config['alphas'])}  |  Strategy: {config['strategy_type']}")
    print("=" * 60)

    # 1. Universe & Panel
    print(f"\n[1] Resolving universe: {config['universe_index']}...")
    symbols = get_index_constituents(config["universe_index"])
    if config["max_stocks"] > 0: symbols = symbols[:config["max_stocks"]]
    print(f"  Universe: {len(symbols)} stocks")

    all_fields = set()
    for a in config["alphas"]: all_fields.update(a.get("data_fields", ["close", "volume"])) # volume and close as default
    all_fields.add("close")
    all_fields_list = sorted(all_fields) # 所有用到的数据字段
    print(f"\n[2] Fetching panel data ({', '.join(all_fields_list)})...")
    panel = build_panel(symbols, config["start_date"], config["end_date"], fields=all_fields_list, verbose=True)
    if "close" not in panel: raise ValueError("Failed to fetch close prices.")
    close = panel["close"]

    print(f"\n[2.5] Fetching risk-free rate...")
    risk_free_daily = get_risk_free_rate(config["start_date"], config["end_date"])
    # print(risk_free_daily)

    # 2.6 Sharpe Benchmark
    sharpe_type = config["sharpe_benchmark"].lower()
    sharpe_benchmark_ret = None
    if sharpe_type == "risk_free":
        sharpe_benchmark_ret = risk_free_daily
    else:
        sharpe_label = sharpe_type.upper()
        print(f"  Fetching Sharpe benchmark: {sharpe_label}...")
        sharpe_benchmark_ret = _fetch_benchmark(sharpe_label, config["start_date"], config["end_date"])

    # 2. Process Each Alpha
    strategy_fn = STRATEGIES.get(config["strategy_type"], _strategy_long_short_zscore)
    all_pnl, all_signals, all_fwd_ret, all_weights = [], [], [], []
    has_neut_factors = False
    combined_neut_factors = []

    for idx, alpha_cfg in enumerate(config["alphas"], 1):
        module_name = alpha_cfg["module"]
        delay, intervalmode, ops = alpha_cfg["delay"], alpha_cfg["intervalmode"], alpha_cfg["ops"]
        print(f"\n[3-{idx}] Alpha: {module_name}  (delay={delay}, interval={intervalmode}, ops={len(ops)})")

        alpha_mod = importlib.import_module(module_name)
        signal = alpha_mod.generate_signal(
            symbols=symbols, start_date=config["start_date"],
            end_date=config["end_date"], verbose=True, panel=panel,
        )
        print(f"  Signal shape: {signal.shape}")

        common_dates = signal.index.intersection(close.index)
        common_tickers = signal.columns.intersection(close.columns)
        signal = signal.loc[common_dates, common_tickers]
        aligned_close = close.loc[common_dates, common_tickers]
        fwd_ret = calc_returns(aligned_close).shift(-delay)
        signal = _apply_intervalmode(signal, intervalmode)

        neut_factors = []
        for o in ops:
            if o.get("type", "").lower() in ("opneutralize", "opriskneut", "neutralize", "risk_neut"):
                neut_factors.extend(o.get("factors", []))
        if neut_factors:
            has_neut_factors = True
            combined_neut_factors.extend(neut_factors)

        signal = apply_op_pipeline(signal, ops, panel=panel, sector_map=None)
        pnl, weights = strategy_fn(signal, fwd_ret, long_short=config["long_short"])
        all_pnl.append(pnl); all_signals.append(signal); all_fwd_ret.append(fwd_ret); all_weights.append(weights)
        print(f"  PnL computed: {len(pnl)} days")

        if config["print_daily_detail"]:
            pnl_clean = pnl.dropna()
            cum = (1 + pnl_clean).cumprod(); peak = cum.cummax()
            hdr = f"{'date':>12s} {'daily_pnl':>10s} {'cum_ret%':>10s} {'sharpe':>8s} {'max_dd%':>8s} {'win%':>7s} {'tdays':>6s}"
            for i, (date, daily) in enumerate(pnl_clean.items()):
                print(hdr)
                n = i + 1
                cr = (cum.iloc[i] - 1) * 100
                dd = ((cum.iloc[i] - peak.iloc[i]) / peak.iloc[i]) * 100
                win = (pnl_clean.iloc[:n] > 0).sum() / n * 100
                sl = pnl_clean.iloc[:n]
                if sharpe_benchmark_ret is not None and len(sharpe_benchmark_ret) > 0:
                    rf_a = sharpe_benchmark_ret.reindex(sl.index).ffill().bfill().fillna(0)
                    exc = sl - rf_a
                    sr = exc.mean() / exc.std() * np.sqrt(252) if n >= 2 and exc.std() != 0 else 0.0
                else:
                    sr = sl.mean() / sl.std() * np.sqrt(252) if n >= 2 and sl.std() != 0 else 0.0
                print(f"{date.strftime('%Y-%m-%d'):>12s} {daily:>10.6f} {cr:>10.4f} {sr:>8.3f} {dd:>8.4f} {win:>7.2f} {n:>6d}")

    # 3. Combine
    if len(all_pnl) == 1:
        pnl_combined, signal_combined, fwd_ret_combined = all_pnl[0], all_signals[0], all_fwd_ret[0]
        weights_combined = all_weights[0]
    else:
        pnl_combined = pd.concat(all_pnl, axis=1).dropna(how="all").mean(axis=1)
        signal_combined, fwd_ret_combined = all_signals[0], all_fwd_ret[0]
        weights_combined = all_weights[0]

    # 4. Transaction Costs
    total_cost_bps = config["commission_bps"] + config["slippage_bps"]
    if total_cost_bps > 0:
        w = signal_combined.div(signal_combined.abs().sum(axis=1), axis=0).fillna(0)
        pnl_combined = pnl_combined - w.diff().abs().sum(axis=1) * (total_cost_bps / 10000.0)
    pnl_combined = pnl_combined.dropna()

    # 5. IR Benchmark
    ir_type = config["ir_benchmark"].lower()
    benchmark_ret, benchmark_label = None, None
    if ir_type == "risk_free":
        benchmark_label = "Risk-Free"
        if risk_free_daily is not None and len(risk_free_daily) > 0:
            benchmark_ret = risk_free_daily.reindex(pnl_combined.index).ffill().bfill().fillna(0)
    else:
        benchmark_label = ir_type.upper()
        print(f"\n[4] Fetching IR benchmark: {benchmark_label}...")
        benchmark_ret = _fetch_benchmark(ir_type.upper(), config["start_date"], config["end_date"])

    # 5.5 Factor Neutralization
    pnl_pre_neut = pnl_combined.copy()
    bm_pre_neut = benchmark_ret.copy() if benchmark_ret is not None else None
    sbm_pre_neut = sharpe_benchmark_ret.copy() if sharpe_benchmark_ret is not None else None
    if has_neut_factors:
        neut_names = sorted(set(combined_neut_factors))
        print(f"\n[5] Factor Neutralization: {neut_names}")
        factor_exposures = compute_factor_exposures(panel, neut_names)
        factor_premiums = compute_factor_premiums(fwd_ret_combined, factor_exposures)
        for fn, fp in factor_premiums.items():
            print(f"  {fn}: {len(fp)} days, mean_premium={fp.mean():.6f}")
        pnl_combined, factor_comp = neutralize_portfolio_returns(
            pnl_combined, weights_combined, factor_exposures, factor_premiums)
        pnl_combined = pnl_combined.dropna()
        if benchmark_ret is not None and config["ir_benchmark"].lower() != "risk_free":
            benchmark_ret = neutralize_benchmark_returns(benchmark_ret, factor_premiums)
        if sharpe_benchmark_ret is not None and config["sharpe_benchmark"].lower() != "risk_free":
            sharpe_benchmark_ret = neutralize_benchmark_returns(sharpe_benchmark_ret, factor_premiums)

    # 6. Results
    print(f"\n{'=' * 60}\n  RESULTS — {name}\n{'=' * 60}")
    stats = compute_stats(pnl_combined, risk_free_daily=sharpe_benchmark_ret, benchmark_ret=benchmark_ret)

    if config["print_stats"]:
        _print_stats_block(f"PERFORMANCE SUMMARY: {name}", stats)

    if config["print_summary_table"]:
        label = f"{name} (neutralised)" if has_neut_factors else name
        _print_summary_table(pnl_combined, label, sharpe_benchmark_ret, benchmark_ret)
    if config["print_summary_table"] and has_neut_factors:
        _print_summary_table(pnl_pre_neut.dropna(), f"{name} (raw)", sbm_pre_neut, bm_pre_neut)
    # Sharpe benchmark label
    sharpe_label = config["sharpe_benchmark"].upper() if config["sharpe_benchmark"].lower() != "risk_free" else "Risk-Free"
    if not no_plot and config["plot_equity_curve"]:
        _plot_equity_curve(pnl_combined, name, benchmark_ret=benchmark_ret, benchmark_label=f"IR Bench ({benchmark_label})",
                           sharpe_benchmark_ret=sharpe_benchmark_ret, sharpe_benchmark_label=f"Sharpe Bench ({sharpe_label})")
    if not no_plot and config["plot_group_returns"]:
        _plot_group_returns(signal_combined, fwd_ret_combined, name)
    if save or config["save_signal_csv"]:
        out_dir = os.path.join(REPO_ROOT, "backtest_main", "output"); os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        signal_combined.to_csv(os.path.join(out_dir, f"{name}_signal_{ts}.csv"))
        pnl_combined.to_csv(os.path.join(out_dir, f"{name}_pnl_{ts}.csv"), header=["daily_pnl"])
        print(f"\n  Saved to {out_dir}")
    if config["save_pnl"]:
        _save_pnl(pnl_combined, name, config, stats)
    return pnl_combined, stats

# ── Output Helpers ──

def _print_stats_block(title, stats):
    print(f"\n{'-' * 40}\n  {title}\n{'-' * 40}")
    for k, v in stats.items():
        label = k.replace("_", " ").title()
        suffix = "%" if any(w in k for w in ("return", "volatility", "drawdown")) else ""
        print(f"  {label:.<30} {v}{suffix}")
    print("-" * 40)

def _compute_period_stats(pnl, risk_free_daily=None, benchmark_ret=None):
    stats = compute_stats(pnl, risk_free_daily=risk_free_daily, benchmark_ret=benchmark_ret)
    if not stats:
        return None
    stats["dates"] = f"{pnl.index[0].strftime('%Y%m%d')}-{pnl.index[-1].strftime('%Y%m%d')}"
    return stats

def _print_summary_table(pnl, name, risk_free_daily=None, benchmark_ret=None):
    print(f"\n\n--- summary: {name} ---")
    header = (f"{'dates':>25s} {'tot%ret':>8s} {'ann%ret':>8s} {'ann%vol':>8s}"
              f" {'sharpe':>8s} {'IR':>8s} {'sortino':>8s} {'calmar':>8s}"
              f" {'%dd':>8s} {'tdays':>6s}")
    print(header)
    def fmt(s):
        return (f"{s['dates']:>25s} {s['total_return']:>8.2f} {s['annual_return']:>8.2f}"
                f" {s['annual_volatility']:>8.2f} {s['sharpe_ratio']:>8.3f}"
                f" {s['information_ratio']:>8.3f} {s['sortino_ratio']:>8.3f}"
                f" {s['calmar_ratio']:>8.3f} {s['max_drawdown']:>8.2f}"
                f" {s['num_trading_days']:>6d}")
    for yr in sorted(pnl.index.year.unique()):
        s = _compute_period_stats(pnl[pnl.index.year == yr], risk_free_daily, benchmark_ret)
        if s: print(fmt(s))
    print()
    total = _compute_period_stats(pnl, risk_free_daily, benchmark_ret)
    if total: print(fmt(total))

# ── Visualization ──

def _plot_equity_curve(pnl, name, benchmark_ret=None, benchmark_label=None,
                       sharpe_benchmark_ret=None, sharpe_benchmark_label=None):
    cum = (1 + pnl).cumprod(); dd = (cum - cum.cummax()) / cum.cummax()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1], sharex=True)
    fig.suptitle(f"Backtest: {name}", fontsize=14, fontweight="bold")
    ax1.plot(cum.index, cum.values, color="#2196F3", linewidth=1.5, label=name)
    if benchmark_ret is not None and benchmark_label:
        bm_aligned = benchmark_ret.reindex(pnl.index).fillna(0)
        bm_cum = (1 + bm_aligned).cumprod()
        ax1.plot(bm_cum.index, bm_cum.values, color="#FF9800", linewidth=1.2, linestyle="--", label=benchmark_label)
    if sharpe_benchmark_ret is not None and sharpe_benchmark_label:
        sb_aligned = sharpe_benchmark_ret.reindex(pnl.index).fillna(0)
        sb_cum = (1 + sb_aligned).cumprod()
        ax1.plot(sb_cum.index, sb_cum.values, color="#4CAF50", linewidth=1.2, linestyle="-.", label=sharpe_benchmark_label)
    ax1.legend(loc="upper left")
    ax1.set_ylabel("Cumulative Return"); ax1.grid(True, alpha=0.3); ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax2.bar(dd.index, dd.values, width=1, color="#F44336", alpha=0.6)
    ax2.set_ylabel("Drawdown"); ax2.set_xlabel("Date"); ax2.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

def _plot_group_returns(signal, fwd_ret, name):
    n_groups = 5
    # Collect per-year, per-group daily returns
    year_group_returns = {}  # {year: {group: [daily_ret, ...]}}
    for date in signal.index:
        yr = date.year
        row = signal.loc[date].dropna(); ret_row = fwd_ret.loc[date].reindex(row.index).dropna()
        common = row.index.intersection(ret_row.index)
        if len(common) < n_groups * 2: continue
        labels = pd.qcut(row[common], n_groups, labels=False, duplicates="drop")
        for g in range(n_groups):
            if (labels == g).sum() > 0:
                year_group_returns.setdefault(yr, {}).setdefault(g, []).append(ret_row[common][labels == g].mean())
    if not year_group_returns:
        return
    years = sorted(year_group_returns.keys())
    q_colors = ["#F44336", "#FF9800", "#9E9E9E", "#4CAF50", "#2196F3"]
    x = np.arange(len(years))
    bar_w = 0.15
    fig, ax = plt.subplots(figsize=(max(10, len(years) * 2), 5))
    for g in range(n_groups):
        vals = [np.mean(year_group_returns[yr].get(g, [0])) * 252 * 100 for yr in years]
        ax.bar(x + g * bar_w, vals, bar_w, label=f"Q{g+1}", color=q_colors[g])
    ax.set_xticks(x + bar_w * (n_groups - 1) / 2)
    ax.set_xticklabels([str(y) for y in years])
    ax.set_ylabel("Annualized Return (%)"); ax.set_title(f"Quintile Returns by Year: {name}")
    ax.legend(loc="best"); ax.grid(True, alpha=0.3, axis="y"); ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout(); plt.show()

# ── PnL Output ──

def _save_pnl(pnl, name, config, stats):
    pnl_dir = os.path.join(REPO_ROOT, "backtest_main", "pnl")
    os.makedirs(pnl_dir, exist_ok=True)
    fpath = os.path.join(pnl_dir, f"{name}.csv")
    pnl_clean = pnl.dropna()
    cum = (1 + pnl_clean).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    lines = []
    for dt, ret in pnl_clean.items():
        c = cum.loc[dt]
        d = dd.loc[dt]
        lines.append(f"{dt.strftime('%Y-%m-%d')},{ret:.8f},{c:.8f},{d:.8f}")
    with open(fpath, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  PnL saved to {fpath} ({len(lines)} days)")

# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="Jade Silhouette — Alpha Backtest Runner")
    parser.add_argument("--config", "-c", required=True, help="Path to backtest XML config")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    config_path = args.config if os.path.isabs(args.config) else os.path.join(REPO_ROOT, args.config)
    if not os.path.exists(config_path):
        print(f"[ERROR] Config not found: {config_path}"); sys.exit(1)
    run_backtest(parse_config(config_path), no_plot=args.no_plot, save=args.save)

if __name__ == "__main__":
    main()
