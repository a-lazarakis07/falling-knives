#!/usr/bin/env python3
"""
backtest.py (BOTTOM ENTRY + FAST PROVE-IT) — FIXED for missing columns

This backtest uses your event file produced by analyze.py:
- Requires: ticker, sector, bottom_date, bottom_close
- Optional: trigger_date, recovery_date, ma50_cross_date

Strategy:
- Entry: bottom_close on bottom_date
- Exit: TP / SL / Prove-it / Time stop

Note: bottom entry is hindsight (good for validating signal quality).
"""

import os
import argparse
import numpy as np
import pandas as pd

EVENTS_PATH_DEFAULT = "data/event_metrics_test.csv"
PRICES_PATH_DEFAULT = "data/prices_dd.parquet"
OUTDIR_DEFAULT = "data"


def parse_args():
    p = argparse.ArgumentParser("Backtest bottom-entry falling-knife strategy (prove-it + TP/SL)")

    p.add_argument("--events", default=EVENTS_PATH_DEFAULT)
    p.add_argument("--prices", default=PRICES_PATH_DEFAULT)
    p.add_argument("--outdir", default=OUTDIR_DEFAULT)

    p.add_argument("--tp-pct", type=float, default=0.12, help="Take profit percent (e.g. 0.12 = +12%)")
    p.add_argument("--sl-pct", type=float, default=0.08, help="Stop loss percent (e.g. 0.08 = -8%)")

    p.add_argument("--prove-days", type=int, default=10, help="Prove-it window in calendar days")
    p.add_argument("--prove-pct", type=float, default=0.02, help="Must be up at least this by prove day or exit")

    p.add_argument("--max-hold-days", type=int, default=63, help="Hard max holding in calendar days")

    return p.parse_args()


def to_dt_if_present(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Load events (don’t assume optional columns exist)
    events = pd.read_csv(args.events)

    if events.empty:
        print("[WARN] No events found in events file.")
        return

    # Convert date columns if present
    for c in ["trigger_date", "bottom_date", "recovery_date", "ma50_cross_date"]:
        to_dt_if_present(events, c)

    # Required core columns
    required = {"ticker", "sector", "bottom_date"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"Events file missing required columns: {missing}")

    # Find bottom price column automatically
    BOTTOM_PRICE_CANDIDATES = [
        "bottom_close",
        "bottom_price",
        "bottom_close_price",
        "bottom_px",
    ]

    bottom_price_col = None
    for c in BOTTOM_PRICE_CANDIDATES:
        if c in events.columns:
            bottom_price_col = c
            break

    if bottom_price_col is None:
        raise ValueError(
            "Could not find bottom price column. "
            "Expected one of: " + ", ".join(BOTTOM_PRICE_CANDIDATES)
        )

    print(f"[INFO] using bottom price column: {bottom_price_col}")

    # Load prices
    prices = pd.read_parquet(args.prices)
    prices["Date"] = pd.to_datetime(prices["Date"])
    if getattr(prices["Date"].dt, "tz", None) is not None:
        prices["Date"] = prices["Date"].dt.tz_convert(None)
    prices = prices.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    prices_by_ticker = {t: df.reset_index(drop=True) for t, df in prices.groupby("Ticker")}

    trades = []

    tp_pct = float(args.tp_pct)
    sl_pct = float(args.sl_pct)

    for _, row in events.iterrows():
        ticker = row["ticker"]
        sector = row["sector"]
        dd_th = row.get("dd_threshold_used", np.nan)

        bottom_date = row["bottom_date"]
        bottom_close = row[bottom_price_col]

        if pd.isna(bottom_date) or pd.isna(bottom_close):
            continue

        df_t = prices_by_ticker.get(ticker)
        if df_t is None or df_t.empty:
            continue

        entry_date = pd.to_datetime(bottom_date)
        entry_price = float(bottom_close)

        deadline = entry_date + pd.Timedelta(days=int(args.max_hold_days))
        fut = df_t[(df_t["Date"] >= entry_date) & (df_t["Date"] <= deadline)].copy()
        if fut.empty:
            continue

        tp_price = entry_price * (1.0 + tp_pct)
        sl_price = entry_price * (1.0 - sl_pct)

        prove_date = entry_date + pd.Timedelta(days=int(args.prove_days))
        fut_prove = fut[fut["Date"] <= prove_date]
        prove_row = fut_prove.iloc[-1] if not fut_prove.empty else None

        exit_reason = "TIME"
        exit_date = fut.iloc[-1]["Date"]
        exit_price = float(fut.iloc[-1]["Close"])

        # walk forward (close-based)
        for r in fut.itertuples():
            px = float(r.Close)

            if px <= sl_price:
                exit_reason = "SL"
                exit_date = r.Date
                exit_price = sl_price
                break

            if px >= tp_price:
                exit_reason = "TP"
                exit_date = r.Date
                exit_price = tp_price
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

        trades.append({
            "ticker": ticker,
            "sector": sector,
            "dd_threshold_used": dd_th,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "holding_days": holding_days,
            "exit_reason": exit_reason,
            "trade_return": trade_return,
            "tp_pct": tp_pct,
            "sl_pct": sl_pct,
            "prove_days": int(args.prove_days),
            "prove_pct": float(args.prove_pct),
        })

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        print("[WARN] No trades generated.")
        return

    out_trades = os.path.join(args.outdir, "trades.csv")
    trades_df.to_csv(out_trades, index=False)
    print(f"[OK] wrote trades -> {out_trades} (n_trades={len(trades_df)})")

    overall = trades_df["trade_return"]
    win_rate = (overall > 0).mean()
    sharpe_like = (overall.mean() / overall.std()) if overall.std() > 0 else np.nan

    print("\n=== Overall Backtest Results (BOTTOM ENTRY + PROVE-IT) ===")
    print(f"TP: +{tp_pct*100:.1f}% | SL: -{sl_pct*100:.1f}% | Prove: +{float(args.prove_pct)*100:.1f}% by day {int(args.prove_days)} | Max hold: {int(args.max_hold_days)}d")
    print(f"Trades: {len(trades_df)} | Win rate: {win_rate:.2%}")
    print(f"Avg return: {overall.mean():.2%} | Median return: {overall.median():.2%}")
    print(f"Avg hold days: {trades_df['holding_days'].mean():.1f} | Median hold days: {trades_df['holding_days'].median():.1f}")
    if not np.isnan(sharpe_like):
        print(f"Trade-level Sharpe-like ratio: {sharpe_like:.2f}")

    print("\nExit reason rates:")
    print(trades_df["exit_reason"].value_counts(normalize=True).mul(100).round(1).astype(str) + "%")

    print("\n=== Sector-level Results ===")
    sector_grp = trades_df.groupby("sector")["trade_return"]
    sector_summary = pd.DataFrame({
        "n_trades": trades_df.groupby("sector")["ticker"].count(),
        "win_rate": sector_grp.apply(lambda s: (s > 0).mean()),
        "avg_return": sector_grp.mean(),
        "median_return": sector_grp.median(),
    }).sort_values("avg_return", ascending=False)

    ss = sector_summary.copy()
    ss["win_rate"] = (ss["win_rate"] * 100).round(1)
    ss["avg_return"] = (ss["avg_return"] * 100).round(2)
    ss["median_return"] = (ss["median_return"] * 100).round(2)
    print(ss.to_string())

    out_sector = os.path.join(args.outdir, "sector_backtest_summary.csv")
    sector_summary.to_csv(out_sector)
    print(f"\n[OK] wrote sector summary -> {out_sector}")



if __name__ == "__main__":
    main()