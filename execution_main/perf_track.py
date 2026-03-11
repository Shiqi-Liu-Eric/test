import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import yaml
import robin_stocks.robinhood as rh

ET = ZoneInfo("America/New_York")
BASE = Path(__file__).resolve().parent
PERF_DIR = BASE / "perf_log"
LOG_DIR = BASE / "trading_log"

def cfg():
    p = BASE.parent / "data_preparation" / "data_apiKey" / "config.yaml"
    with open(p) as f:
        return yaml.safe_load(f)

def login():
    c = cfg()
    rh.login(username=c["keys"]["robinhood_username"],
             password=c["keys"]["robinhood_password"])

def get_holdings():
    raw = rh.account.build_holdings()
    rows = []
    for tk, info in raw.items():
        qty = float(info.get("quantity", 0))
        if qty == 0:
            continue
        avg_cost = float(info.get("average_buy_price", 0))
        px = float(info.get("price", 0))
        mv = float(info.get("equity", 0))
        rows.append({
            "ticker": tk, "quantity": qty, "avg_cost": avg_cost,
            "market_price": px, "market_value": mv,
        })
    return pd.DataFrame(rows)

def prev_snapshot(today_str):
    files = sorted(PERF_DIR.glob("*.csv"))
    for f in reversed(files):
        if f.stem < today_str:
            try:
                df = pd.read_csv(f, nrows=50)
                df = df[df["ticker"].notna() & (df["ticker"] != "_TOTAL")]
                return df.set_index("ticker")
            except Exception:
                continue
    return None

def calc_daily_pnl(holdings, prev):
    if prev is None or holdings.empty:
        holdings["daily_pnl"] = 0.0
        holdings["total_pnl"] = (holdings["market_price"] - holdings["avg_cost"]) * holdings["quantity"]
        return holdings

    holdings = holdings.copy()
    daily, total = [], []
    for _, r in holdings.iterrows():
        tk = r["ticker"]
        cost = r["avg_cost"]
        qty = r["quantity"]
        px_now = r["market_price"]

        if tk in prev.index and tk != "_CASH":
            px_prev = prev.loc[tk, "market_price"]
            prev_qty = prev.loc[tk, "quantity"]
            d_pnl = (px_now - px_prev) * min(qty, prev_qty)
            if qty > prev_qty:
                d_pnl += (px_now - cost) * (qty - prev_qty)
        else:
            d_pnl = (px_now - cost) * qty

        t_pnl = (px_now - cost) * qty
        daily.append(d_pnl)
        total.append(t_pnl)

    holdings["daily_pnl"] = daily
    holdings["total_pnl"] = total
    return holdings

def build_snapshot(today_str):
    holdings = get_holdings()

    profile = rh.profiles.load_portfolio_profile()
    equity = float(profile.get("equity", 0))
    cash = float(profile.get("withdrawable_amount", 0) or 0)
    try:
        acct = rh.profiles.load_account_profile()
        cash = float(acct.get("cash", cash) or cash)
    except Exception:
        pass

    prev = prev_snapshot(today_str)

    if prev is not None and "_TOTAL" not in prev.index:
        aum_open = equity - holdings["market_value"].sum() + \
                   (prev["market_value"].sum() if not prev.empty else 0) + cash
    else:
        aum_open = equity

    holdings = calc_daily_pnl(holdings, prev)
    aum_mv = holdings["market_value"].sum()
    holdings["weight"] = holdings["market_value"] / equity if equity > 0 else 0

    cash_row = pd.DataFrame([{
        "ticker": "_CASH", "quantity": 1, "avg_cost": cash,
        "market_price": cash, "market_value": cash,
        "daily_pnl": 0.0, "total_pnl": 0.0,
        "weight": cash / equity if equity > 0 else 0,
    }])

    total_daily = holdings["daily_pnl"].sum()
    total_total = holdings["total_pnl"].sum()
    total_row = pd.DataFrame([{
        "ticker": "_TOTAL", "quantity": np.nan, "avg_cost": np.nan,
        "market_price": np.nan, "market_value": np.nan,
        "daily_pnl": total_daily, "total_pnl": total_total,
        "weight": 1.0,
    }])

    df = pd.concat([holdings, cash_row, total_row], ignore_index=True)
    df.insert(0, "date", today_str[:4] + "-" + today_str[4:6] + "-" + today_str[6:])

    daily_ret = total_daily / aum_open if aum_open > 0 else 0.0
    cum_ret = equity / aum_open - 1 if aum_open > 0 and aum_open != equity else daily_ret

    prev_files = sorted(PERF_DIR.glob("*.csv"))
    if prev_files:
        try:
            last = pd.read_csv(prev_files[-1], skiprows=lambda i: i < _find_summary_row(prev_files[-1]))
            prev_cum = float(last.iloc[0].get("cumulative_return", 0))
            cum_ret = (1 + prev_cum) * (1 + daily_ret) - 1
        except Exception:
            pass

    summary = pd.DataFrame([{
        "date": df["date"].iloc[0],
        "aum_open": round(aum_open, 2),
        "aum_close": round(equity, 2),
        "daily_return": round(daily_ret, 6),
        "cumulative_return": round(cum_ret, 6),
    }])

    return df, summary


def _find_summary_row(filepath):
    with open(filepath) as f:
        for i, line in enumerate(f):
            if line.startswith("date,aum_open"):
                return i
    return 0


def save(df, summary, today_str):
    PERF_DIR.mkdir(parents=True, exist_ok=True)
    out = PERF_DIR / f"{today_str}.csv"
    df.to_csv(out, index=False, float_format="%.2f")
    with open(out, "a") as f:
        f.write(summary.to_csv(index=False, float_format="%.6f"))
    return out

def main():
    if len(sys.argv) >= 2:
        today = sys.argv[1]
    else:
        today = datetime.now(ET).strftime("%Y%m%d")

    print(f"perf_track [{today}]")

    login()

    df, summary = build_snapshot(today)

    out = save(df, summary, today)

    print(f"\n{df.to_string(index=False)}")
    print(f"\n{summary.to_string(index=False)}")
    print(f"\n→ {out}")

    rh.logout()

if __name__ == "__main__":
    main()
