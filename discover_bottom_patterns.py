#!/usr/bin/env python3
"""
backtest_confirm.py — tradable bottom-proxy entry from trigger

ENTRY (tradable):
- Start at trigger_date.
- Maintain running_low = minimum Close seen since trigger.
- Record low_date when running_low updates.
- Only allow entry if:
    1) Close >= running_low * (1 + confirm_pct)
    2) We are at least min_days_since_low calendar days after low_date
    3) Optional: require no_new_low_days: lowest Close in last N days >= running_low
       (i.e., low hasn't been broken recently)

EXIT (same as your benchmark style):
- TP at +tp_pct
- SL at -sl_pct
- Prove-it: by prove_days must be up prove_pct else exit
- Max hold: max_hold_days calendar days

Inputs:
- events file must contain: ticker, sector, trigger_date
- prices_dd.parquet must contain: Date, Ticker, Close
"""

import os
import argparse
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser("Backtest falling-knife with tradable bottom-confirmation entry")

    p.add_argument("--events", default="data/event_metrics_test.csv")
    p.add_argument("--prices", default="data/prices_dd.parquet")
    p.add_argument("--outdir", default="data")

    # Entry: bottom confirmation
    p.add_argument("--entry-wait-days", type=int, default=45, help="How long after trigger we search for an entry")
    p.add_argument("--confirm-pct", type=float, default=0.06, help="Enter after price is X%% above running low")
    p.add_argument("--min-days-since-low", type=int, default=2, help="Require at least N calendar days after low_date")
    p.add_argument("--no-new-low-days", type=int, default=3, help="Require no new low in last N days (0 disables)")

    # Exits
    p.add_argument("--tp-pct", type=float, default=0.08)
    p.add_argument("--sl-pct", type=float, default=0.10)
    p.add_argument("--prove-days", type=int, default=10)
    p.add_argument("--prove-pct", type=float, default=0.01)
    p.add_argument("--max-hold-days", type=int, default=63)

    # Optional sector allowlist (super picky)
    p.add_argument("--sector-allow", nargs="*", default=None,
                   help="If provided, only trade these sectors (exact names).")

    return p.parse_args()


def to_dt(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")


def run_exit(fut: pd.DataFrame, entry_date: pd.Timestamp, entry_price: float, args):
    tp = entry_price * (1.0 + float(args.tp_pct))
    sl = entry_price * (1.0 - float(args.sl_pct))

    prove_date = entry_date + pd.Timedelta(days=int(args.prove_days))
    fut_prove = fut[fut["Date"] <= prove_date]
    prove_row = fut_prove.iloc[-1] if not fut_prove.empty else None

    exit_reason = "TIME"
    exit_date = fut.iloc[-1]["Date"]
    exit_price = float(fut.iloc[-1]["Close"])

    for r in fut.itertuples():
        px = float(r.Close)

        if px <= sl:
            exit_reason = "SL"
            exit_date = r.Date
            exit_price = sl
            break

        if px >= tp:
            exit_reason = "TP"
            exit_date = r.Date
            exit_price = tp
            break

        if prove_row is not None and r.Date == prove_row.Date:
            prove_ret = (float(prove_row.Close) / entry_price) - 1.0
            if prove_ret < float(args.prove_pct):
                exit_reason = "PROVE_FAIL"
                exit_date = prove_row.Date
                exit_price = float(prove_row.Close)
                break

    holding_days = int((pd.to_datetime(exit_date) - entry_date).days)
    trade_return = (exit_price / entry_price) - 1.0
    return exit_date, exit_price, exit_reason, holding_days, trade_return


def find_entry_confirm(df_t: pd.DataFrame, trigger_date: pd.Timestamp, args):
    """
    Returns (entry_date, entry_price, low_date, running_low) or None.
    """
    end_date = trigger_date + pd.Timedelta(days=int(args.entry_wait_days))
    w = df_t[(df_t["Date"] >= trigger_date) & (df_t["Date"] <= end_date)].copy()
    if w.empty:
        return None

    running_low = np.inf
    low_date = None

    closes = w["Close"].astype(float).values
    dates = w["Date"].values

    # We'll step day by day
    for i in range(len(w)):
        d = pd.to_datetime(dates[i])
        px = float(closes[i])

        # update running low
        if px < running_low:
            running_low = px
            low_date = d

        if low_date is None:
            continue

        # require some time since the low
        if (d - low_date).days < int(args.min_days_since_low):
            continue

        # optional: no new low recently
        if int(args.no_new_low_days) > 0:
            look = w[(w["Date"] <= d) & (w["Date"] >= d - pd.Timedelta(days=int(args.no_new_low_days)))]
            if not look.empty:
                recent_min = float(look["Close"].astype(float).min())
                if recent_min < running_low:
                    # should never happen because running_low is min since trigger,
                    # but keep for safety.
                    continue

        # confirmation threshold
        if px >= running_low * (1.0 + float(args.confirm_pct)):
            return d, px, low_date, running_low

    return None


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    events = pd.read_csv(args.events)
    to_dt(events, "trigger_date")

    required = {"ticker", "sector", "trigger_date"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"Events file missing required columns: {missing}")

    prices = pd.read_parquet(args.prices)
    prices["Date"] = pd.to_datetime(prices["Date"])
    if getattr(prices["Date"].dt, "tz", None) is not None:
        prices["Date"] = prices["Date"].dt.tz_convert(None)
    prices = prices.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    prices_by_ticker = {t: df.reset_index(drop=True) for t, df in prices.groupby("Ticker")}

    trades = []
    skipped_sector = 0
    skipped_no_prices = 0
    skipped_no_entry = 0

    allow = set(args.sector_allow) if args.sector_allow else None

    for _, ev in events.iterrows():
        ticker = ev["ticker"]
        sector = ev["sector"]
        trig = ev["trigger_date"]

        if pd.isna(trig):
            continue

        if allow is not None and sector not in allow:
            skipped_sector += 1
            continue

        df_t = prices_by_ticker.get(ticker)
        if df_t is None or df_t.empty:
            skipped_no_prices += 1
            continue

        entry_info = find_entry_confirm(df_t, pd.to_datetime(trig), args)
        if entry_info is None:
            skipped_no_entry += 1
            continue

        entry_date, entry_price, low_date, running_low = entry_info

        # exit window
        deadline = entry_date + pd.Timedelta(days=int(args.max_hold_days))
        fut = df_t[(df_t["Date"] >= entry_date) & (df_t["Date"] <= deadline)].copy()
        if fut.empty:
            skipped_no_prices += 1
            continue

        exit_date, exit_price, exit_reason, holding_days, trade_return = run_exit(
            fut, entry_date, float(entry_price), args
        )

        trades.append({
            "ticker": ticker,
            "sector": sector,
            "trigger_date": pd.to_datetime(trig),
            "low_date": low_date,
            "running_low": float(running_low),
            "entry_date": entry_date,
            "entry_price": float(entry_price),
            "exit_date": exit_date,
            "exit_price": float(exit_price),
            "exit_reason": exit_reason,
            "holding_days": holding_days,
            "trade_return": float(trade_return),
            "confirm_pct": float(args.confirm_pct),
            "min_days_since_low": int(args.min_days_since_low),
            "no_new_low_days": int(args.no_new_low_days),
        })

    trades_df = pd.DataFrame(trades)
    out_trades = os.path.join(args.outdir, "trades_confirm.csv")
    trades_df.to_csv(out_trades, index=False)
    print(f"[OK] wrote {out_trades} (n_trades={len(trades_df)})")
    print(f"[INFO] skipped_sector={skipped_sector}, skipped_no_prices={skipped_no_prices}, skipped_no_entry={skipped_no_entry}")

    if trades_df.empty:
        print("[WARN] No trades generated. Try relaxing confirm rules (lower confirm_pct, fewer days).")
        return

    r = trades_df["trade_return"]
    win_rate = (r > 0).mean()
    sharpe_like = (r.mean() / r.std()) if r.std() > 0 else np.nan

    print("\n=== Overall Results (CONFIRM FROM RUNNING LOW) ===")
    print(f"Entry: +{float(args.confirm_pct)*100:.1f}% off running low, min {int(args.min_days_since_low)}d since low, no-new-low {int(args.no_new_low_days)}d")
    print(f"Exits: TP +{float(args.tp_pct)*100:.1f}% | SL -{float(args.sl_pct)*100:.1f}% | Prove +{float(args.prove_pct)*100:.1f}% by {int(args.prove_days)}d | Max hold {int(args.max_hold_days)}d")
    print(f"Trades: {len(trades_df)} | Win rate: {win_rate:.2%}")
    print(f"Avg return: {r.mean():.2%} | Median return: {r.median():.2%}")
    print(f"Sharpe-like: {sharpe_like:.2f}")

    print("\nExit reason rates:")
    print(trades_df["exit_reason"].value_counts(normalize=True).mul(100).round(1).astype(str) + "%")

    # sector summary
    out_sector = os.path.join(args.outdir, "sector_confirm_summary.csv")
    sector_grp = trades_df.groupby("sector")["trade_return"]
    sector_summary = pd.DataFrame({
        "n_trades": trades_df.groupby("sector")["ticker"].count(),
        "win_rate": sector_grp.apply(lambda s: (s > 0).mean()),
        "avg_return": sector_grp.mean(),
        "median_return": sector_grp.median(),
    }).sort_values("avg_return", ascending=False)
    sector_summary.to_csv(out_sector)
    print(f"\n[OK] wrote {out_sector}")


if __name__ == "__main__":
    main()
