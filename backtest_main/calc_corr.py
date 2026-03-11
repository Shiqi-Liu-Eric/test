"""calc_corr.py — Compute PnL correlation between a given alpha and all others.
Usage: python backtest_main/calc_corr.py <alpha_name>
Example: python backtest_main/calc_corr.py alpha_wq008
"""
import os, sys, glob
import numpy as np
import pandas as pd

PNL_DIR = os.path.join(os.path.dirname(__file__), "pnl")


def load_pnl(name):
    fpath = os.path.join(PNL_DIR, f"{name}.csv")
    if not os.path.exists(fpath):
        return None
    df = pd.read_csv(fpath, header=None, names=["date", "ret", "cum", "dd"])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["ret"]


def main():
    if len(sys.argv) < 2:
        print("Usage: python backtest_main/calc_corr.py <alpha_name>")
        sys.exit(1)

    target = sys.argv[1]
    target_pnl = load_pnl(target)
    if target_pnl is None:
        print(f"[ERROR] PnL file not found: {os.path.join(PNL_DIR, target + '.csv')}")
        sys.exit(1)

    files = glob.glob(os.path.join(PNL_DIR, "*.csv"))
    corrs = {}
    for f in files:
        other_name = os.path.splitext(os.path.basename(f))[0]
        if other_name == target:
            continue
        other_pnl = load_pnl(other_name)
        if other_pnl is None:
            continue
        common = target_pnl.index.intersection(other_pnl.index)
        if len(common) < 30:
            continue
        a = target_pnl.loc[common].fillna(0)
        b = other_pnl.loc[common].fillna(0)
        c = a.corr(b)
        if np.isnan(c):
            continue
        corrs[other_name] = (c, len(common))

    if not corrs:
        print(f"No other alphas with sufficient overlapping dates found for {target}.")
        return

    sorted_corrs = sorted(corrs.items(), key=lambda x: x[1][0], reverse=True)

    print(f"\n{'=' * 50}")
    print(f"  Correlation Report: {target}")
    print(f"  Target PnL: {len(target_pnl)} days")
    print(f"  Compared against: {len(corrs)} alphas")
    print(f"{'=' * 50}")

    n_show = min(5, len(sorted_corrs))

    print(f"\n  Top {n_show} (highest correlation):")
    print(f"  {'alpha':<25s} {'corr':>8s} {'overlap':>8s}")
    for name, (c, n) in sorted_corrs[:n_show]:
        print(f"  {name:<25s} {c:>8.4f} {n:>8d}")

    print(f"\n  Bottom {n_show} (lowest correlation):")
    print(f"  {'alpha':<25s} {'corr':>8s} {'overlap':>8s}")
    for name, (c, n) in sorted_corrs[-n_show:]:
        print(f"  {name:<25s} {c:>8.4f} {n:>8d}")

    print()


if __name__ == "__main__":
    main()
