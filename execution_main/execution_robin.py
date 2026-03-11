# execution_robin.py — Robinhood API Execution (via robin_stocks)
#
# ============================================================
# Supported Order Types (robin_stocks.robinhood.orders)
# ============================================================
#
# --- Stock Orders ---
#   order_buy_market()          Market buy
#   order_sell_market()         Market sell
#   order_buy_limit()           Limit buy
#   order_sell_limit()          Limit sell
#   order_buy_stop_loss()       Stop-loss buy
#   order_sell_stop_loss()      Stop-loss sell
#   order_buy_stop_limit()      Stop-limit buy
#   order_sell_stop_limit()     Stop-limit sell
#   order_buy_trailing_stop()   Trailing-stop buy
#   order_sell_trailing_stop()  Trailing-stop sell
#   order()                     Generic (custom trigger/orderType)
#
# --- Option Orders ---
#   order_buy_option_limit()        Limit buy option
#   order_sell_option_limit()       Limit sell option
#   order_buy_option_stop_limit()   Stop-limit buy option
#   order_sell_option_stop_limit()  Stop-limit sell option
#   order_option_spread()           Option spread (generic)
#   order_option_credit_spread()    Credit spread
#   order_option_debit_spread()     Debit spread
#
# --- Crypto Orders ---
#   order_buy_crypto_by_price()         Buy crypto by dollar amount
#   order_buy_crypto_by_quantity()      Buy crypto by quantity
#   order_buy_crypto_limit()            Limit buy crypto
#   order_buy_crypto_limit_by_price()   Limit buy crypto by dollar amount
#   order_sell_crypto_by_price()        Sell crypto by dollar amount
#   order_sell_crypto_by_quantity()      Sell crypto by quantity
#   order_sell_crypto_limit()           Limit sell crypto
#   order_sell_crypto_limit_by_price()  Limit sell crypto by dollar amount
#
# ============================================================

import sys
import sys, math, time
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import robin_stocks.robinhood as rh

ET = ZoneInfo("America/New_York")
MOC_SUBMIT_TIME = (15, 55)  # submit MOC orders at 15:55 ET

STRATEGY_ALLOC = {
    "pureAlpha": 0.25, "indexEnhanced": 0.10,
    "allWeather": 0.15, "riskParity": 0.15,
    "ctaDerivatives": 0.05, "mnaArbitrage": 0.05,
    "usTreasury": 0.05, "optionTrade": 0.05,
}
CASH_ALLOC = 0.30

FILE_TO_STRAT = {
    "strategy_pureAlpha_weight.csv": "pureAlpha",
    "strategy_indexEnhanced_weight.csv": "indexEnhanced",
    "strategy_allWeather_weight.csv": "allWeather",
    "strategy_riskParity_weight.csv": "riskParity",
    "strategy_ctaDerivatives_weight.csv": "ctaDerivatives",
    "strategy_mnaArbitrage_weight.csv": "mnaArbitrage",
    "strategy_usTreasury_weight.csv": "usTreasury",
    "strategy_optionTrade_weight.csv": "optionTrade",
}

ORDER_FUNCS = {
    "default":    {"buy": rh.orders.order_buy_market,     "sell": rh.orders.order_sell_market},
    "market":     {"buy": rh.orders.order_buy_market,     "sell": rh.orders.order_sell_market},
    "limit":      {"buy": rh.orders.order_buy_limit,      "sell": rh.orders.order_sell_limit},
    "stop_loss":  {"buy": rh.orders.order_buy_stop_loss,  "sell": rh.orders.order_sell_stop_loss},
    "stop_limit": {"buy": rh.orders.order_buy_stop_limit, "sell": rh.orders.order_sell_stop_limit},
}

def load_config() -> dict:
    cfg_path = Path(__file__).resolve().parent.parent / "data_preparation" / "data_apiKey" / "config.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def resolve_alloc(found_strats: set) -> dict:
    has_aw = "allWeather" in found_strats
    has_rp = "riskParity" in found_strats
    alloc = {}
    for s in found_strats:
        base = STRATEGY_ALLOC.get(s, 0.0)
        if s == "allWeather":
            alloc[s] = base / 2 if has_rp else base
        elif s == "riskParity":
            alloc[s] = base / 2 if has_aw else base
        else:
            alloc[s] = base
    return alloc


def read_strategy_csvs(log_dir: Path) -> dict:
    strats = {}
    for f in sorted(log_dir.glob("strategy_*_weight.csv")):
        key = FILE_TO_STRAT.get(f.name)
        if key is None:
            print(f"  [WARN] unrecognised: {f.name}")
            continue
        strats[key] = pd.read_csv(f)
        print(f"  ✓ {f.name} → {key} ({len(strats[key])} rows)")
    return strats


def build_meta_weights(strats: dict, alloc: dict) -> pd.DataFrame:
    rows = []
    for strat, df in strats.items():
        w = alloc.get(strat, 0.0)
        for _, r in df.iterrows():
            rows.append({
                "ticker": r["ticker"], "meta_weight": w * r["weight"],
                "order_time": r.get("order_time", "MOC"),
                "order_type": r.get("order_type", "default"), "strategy": strat,
            })
    if not rows:
        return pd.DataFrame(columns=["ticker", "meta_weight", "order_time", "order_type"])

    raw = pd.DataFrame(rows)
    agg = raw.groupby("ticker")["meta_weight"].sum()
    idx_dom = raw.groupby("ticker")["meta_weight"].apply(lambda s: s.abs().idxmax())
    dom = raw.loc[idx_dom.values, ["ticker", "order_time", "order_type"]].set_index("ticker")
    meta = pd.DataFrame({"meta_weight": agg}).join(dom).reset_index()
    return meta.sort_values("meta_weight", ascending=False, key=abs).reset_index(drop=True)


def get_current_positions() -> dict:
    holdings = rh.account.build_holdings()
    return {tk: float(info.get("quantity", 0)) for tk, info in holdings.items()
            if float(info.get("quantity", 0)) != 0}


def get_latest_price(ticker: str) -> float:
    try:
        q = rh.stocks.get_latest_price(ticker)
        if q and q[0] is not None:
            return float(q[0])
    except Exception:
        pass
    return 0.0


def compute_trade_list(meta: pd.DataFrame, current_pos: dict, invest_value: float) -> pd.DataFrame:
    trades = []
    for _, row in meta.iterrows():
        ticker = row["ticker"]
        target_dollar = row["meta_weight"] * invest_value
        price = get_latest_price(ticker)
        if price <= 0:
            print(f"  [WARN] no price for {ticker}")
            continue
        target_sh = int(np.sign(target_dollar) * math.floor(abs(target_dollar) / price))
        cur_sh = current_pos.pop(ticker, 0)
        delta = target_sh - cur_sh
        if delta == 0:
            continue
        trades.append({
            "ticker": ticker, "price": price,
            "target_shares": target_sh, "current_shares": cur_sh,
            "delta_shares": abs(delta), "side": "buy" if delta > 0 else "sell",
            "order_time": row["order_time"], "order_type": row["order_type"],
        })
    for ticker, qty in current_pos.items():
        if qty == 0:
            continue
        trades.append({
            "ticker": ticker, "price": get_latest_price(ticker),
            "target_shares": 0, "current_shares": qty,
            "delta_shares": abs(qty), "side": "sell",
            "order_time": "DAY", "order_type": "default",
        })
    return pd.DataFrame(trades)


def _submit_order(ticker, side, qty, order_type, tif="gfd", extended=False, dry_run=False):
    funcs = ORDER_FUNCS.get(order_type, ORDER_FUNCS["default"])
    fn = funcs[side]
    tag = f"  {side.upper():4s} {qty:>6d} {ticker:<8s} type={order_type} tif={tif} ext={extended}"
    print(tag)
    if dry_run:
        return
    try:
        kwargs = {"symbol": ticker, "quantity": qty, "timeInForce": tif}
        if extended:
            kwargs["extendedHours"] = True
        result = fn(**kwargs)
        oid = result.get("id", "n/a") if isinstance(result, dict) else "n/a"
        print(f"       → order id={oid}")
    except Exception as e:
        print(f"       → ERROR: {e}")


def _wait_until_et(hour, minute):
    """Block until the given ET wall-clock time (today)."""
    now = datetime.now(ET)
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        return
    wait = (target - now).total_seconds()
    print(f"  ⏳ waiting {wait/60:.1f} min until {hour:02d}:{minute:02d} ET …")
    time.sleep(wait)


def execute_trades(trade_df: pd.DataFrame, dry_run: bool = False):
    if trade_df.empty:
        print("\n✓ No trades — portfolio on target.")
        return

    moo = trade_df[trade_df["order_time"] == "MOO"]
    moc = trade_df[trade_df["order_time"] == "MOC"]
    intra = trade_df[~trade_df["order_time"].isin(["MOO", "MOC"])]

    if not moo.empty:
        print(f"\n── MOO orders ({len(moo)}) — extendedHours pre-market ──")
        for _, t in moo.iterrows():
            _submit_order(t["ticker"], t["side"], int(t["delta_shares"]),
                          t["order_type"], tif="gfd", extended=True, dry_run=dry_run)
    if not intra.empty:
        print(f"\n── Intraday orders ({len(intra)}) — gfd ──")
        for _, t in intra.iterrows():
            tif = "gtc" if t["order_time"] == "GTC" else "gfd"
            _submit_order(t["ticker"], t["side"], int(t["delta_shares"]),
                          t["order_type"], tif=tif, extended=False, dry_run=dry_run)
    if not moc.empty:
        print(f"\n── MOC orders ({len(moc)}) — deferred to {MOC_SUBMIT_TIME[0]}:{MOC_SUBMIT_TIME[1]:02d} ET ──")
        if not dry_run:
            _wait_until_et(*MOC_SUBMIT_TIME)
        for _, t in moc.iterrows():
            _submit_order(t["ticker"], t["side"], int(t["delta_shares"]),
                          t["order_type"], tif="gfd", extended=False, dry_run=dry_run)

def main():
    if len(sys.argv) < 2:
        print("Usage: python execution_robin.py <YYYYMMDD> [--dry-run]")
        sys.exit(1)

    date_str = sys.argv[1]
    dry_run = "--dry-run" in sys.argv

    try:
        datetime.strptime(date_str, "%Y%m%d")
    except ValueError:
        print(f"ERROR: invalid date '{date_str}'")
        sys.exit(1)

    log_dir = Path(__file__).resolve().parent / "trading_log" / date_str
    if not log_dir.exists():
        print(f"ERROR: {log_dir} not found")
        sys.exit(1)

    print(f"\n{'─'*60}")
    print(f"  Jade Silhouette — Robinhood Execution  [{date_str}]")
    print(f"{'─'*60}")

    cfg = load_config()
    rh_user = cfg["keys"]["robinhood_username"]
    rh_pass = cfg["keys"]["robinhood_password"]
    if not rh_user or not rh_pass:
        print("ERROR: robinhood credentials not set in config.yaml")
        sys.exit(1)

    print("\n[1] Login")
    rh.login(username=rh_user, password=rh_pass)
    equity = rh.profiles.load_portfolio_profile().get("equity")
    portfolio_value = float(equity) if equity else 0.0
    invest_value = portfolio_value * (1 - CASH_ALLOC)
    print(f"    portfolio=${portfolio_value:,.2f}  cash30%=${portfolio_value*CASH_ALLOC:,.2f}  invest70%=${invest_value:,.2f}")

    print(f"\n[2] Read strategy CSVs from {log_dir.name}/")
    strats = read_strategy_csvs(log_dir)
    if not strats:
        print("ERROR: no CSVs found"); sys.exit(1)

    print(f"\n[3] Build meta weights")
    alloc = resolve_alloc(set(strats.keys()))
    for s, a in sorted(alloc.items(), key=lambda x: -x[1]):
        print(f"    {s:<20s} {a:.1%}")
    meta = build_meta_weights(strats, alloc)
    w_sum = meta["meta_weight"].abs().sum()
    if w_sum > 0:
        meta["meta_weight"] = meta["meta_weight"] / w_sum * (1 - CASH_ALLOC)
    print(f"    {len(meta)} assets")
    print(meta.to_string(index=False, float_format="%.6f"))
    meta.to_csv(log_dir / "meta_weight.csv", index=False)

    print(f"\n[4] Current positions")
    current_pos = get_current_positions()
    for tk, qty in sorted(current_pos.items()):
        print(f"    {tk:<8s} {qty:>10.2f}")
    if not current_pos:
        print("    (none)")

    print(f"\n[5] Compute trades")
    trade_df = compute_trade_list(meta, current_pos, invest_value)
    if not trade_df.empty:
        print(trade_df.to_string(index=False, float_format="%.2f"))
        trade_df.to_csv(log_dir / "trade_list.csv", index=False)
    else:
        print("    ✓ on target")

    print(f"\n[6] Execute {'[DRY RUN]' if dry_run else ''}")
    execute_trades(trade_df, dry_run=dry_run)

    print(f"\n{'─'*60}")
    print(f"  Done. {'(DRY RUN)' if dry_run else ''}")
    print(f"{'─'*60}\n")
    rh.logout()

if __name__ == "__main__":
    main()