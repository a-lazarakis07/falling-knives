#!/usr/bin/env python3
"""
run_best_strategy.py

Run the already-selected best strategy config (from optimize_strategy.py)
WITHOUT doing any optimization or train/test selection.

Inputs:
- data/best_train_config.json  (contains filter, tp, sl, hold)
- events CSV (from analyze.py; can be any date window you want)
- prices parquet (prices_dd.parquet)

Outputs:
- data/run_trades.csv
- data/run_metrics.csv
"""

import os
import json
import argparse
import numpy as np
import pandas as pd

TRANSACTION_COST = 0.01  # 1% round-trip cost

def parse_args():
    p = argparse.ArgumentParser("Run frozen best strategy config (no re-optimization)")
    p.add_argument("--config", default="data/best_train_config.json", help="Best config JSON produced by optimize_strategy.py")
    p.add_argument("--events", default="data/event_metrics_test.csv", help="Events CSV produced by analyze.py")
    p.add_argument("--prices", default="data/prices_dd.parquet", help="Prepared prices parquet")
    p.add_argument("--outdir", default="data")
    p.add_argument("--start", default=None, help="Optional: only keep events with trigger_date >= start (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="Optional: only keep events with trigger_date <= end (YYYY-MM-DD)")
    return p.parse_args()


def normalize_dates(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce")
    try:
        s = s.dt.tz_localize(None)
    except Exception:
        pass
    return s.dt.normalize()


def build_filters(events: pd.DataFrame) -> dict:
    """Must match optimize_strategy.py filter definitions."""
    trigger_counts = events.groupby("ticker").size()

    return {
        "ALL": events,
        "TOP_3_SECTORS": events[events["sector"].isin(["Energy", "Utilities", "Financial Services"])],
        "ENERGY_ONLY": events[events["sector"] == "Energy"],
        "UTILITIES_ONLY": events[events["sector"] == "Utilities"],
        "FINANCIAL_ONLY": events[events["sector"] == "Financial Services"],
        "LOW_TRIGGER": events[events["ticker"].isin(trigger_counts[trigger_counts <= 5].index)],
        "MED_TRIGGER": events[events["ticker"].isin(trigger_counts[(trigger_counts > 5) & (trigger_counts <= 10)].index)],
        "TOP_SECTORS_LOW_TRIGGER": events[
            events["sector"].isin(["Energy", "Utilities", "Financial Services"]) &
            events["ticker"].isin(trigger_counts[trigger_counts <= 5].index)
        ],
        "NO_TECH": events[events["sector"] != "Technology"],
        "NO_TECH_NO_REALESTATE": events[~events["sector"].isin(["Technology", "Real Estate"])],
    }


def run_backtest(events: pd.DataFrame, prices_by_ticker: dict,
                 tp_pct: float, sl_pct: float, max_hold_days: int) -> pd.DataFrame:
    """Entry next trading day >= trigger_date. Exit TP/SL/TIME."""
    trades = []

    for _, row in events.iterrows():
        ticker = row["ticker"]
        sector = row.get("sector", "Unknown")
        trigger_date = row["trigger_date"]

        if pd.isna(trigger_date):
            continue

        df_t = prices_by_ticker.get(ticker)
        if df_t is None or df_t.empty:
            continue

        # entry on next available trading day on/after trigger_date
        idx = df_t["Date"].searchsorted(trigger_date)
        if idx >= len(df_t):
            continue

        entry_date = pd.to_datetime(df_t.iloc[idx]["Date"])
        entry_price = float(df_t.iloc[idx]["Close"])

        deadline = entry_date + pd.Timedelta(days=int(max_hold_days))
        fut = df_t[(df_t["Date"] > entry_date) & (df_t["Date"] <= deadline)].copy()
        if fut.empty:
            continue

        tp_price = entry_price * (1.0 + float(tp_pct))
        sl_price = entry_price * (1.0 - float(sl_pct))

        exit_reason = "TIME"
        exit_date = pd.to_datetime(fut.iloc[-1]["Date"])
        exit_price = float(fut.iloc[-1]["Close"])

        for r in fut.itertuples():
            px = float(r.Close)
            if px <= sl_price:
                exit_reason = "SL"
                exit_date = pd.to_datetime(r.Date)
                exit_price = sl_price
                break
            if px >= tp_price:
                exit_reason = "TP"
                exit_date = pd.to_datetime(r.Date)
                exit_price = tp_price
                break

        trades.append({
            "ticker": ticker,
            "sector": sector,
            "trigger_date": trigger_date,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "holding_days": int((exit_date - entry_date).days),
            "exit_reason": exit_reason,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "gross_return": (exit_price / entry_price) - 1.0,
            "trade_return": ((exit_price / entry_price) - 1.0) - TRANSACTION_COST
        })

    return pd.DataFrame(trades)


def analyze_trades(trades: pd.DataFrame) -> dict | None:
    if trades is None or trades.empty:
        return None

    r = trades["trade_return"].astype(float)
    win_rate = float((r > 0).mean())
    avg_return = float(r.mean())
    med_return = float(r.median())
    std = float(r.std(ddof=0))
    sharpe = float(avg_return / std) if std > 0 else 0.0

    wins = float(r[r > 0].sum())
    losses = float(abs(r[r < 0].sum()))
    profit_factor = float(wins / losses) if losses > 0 else float("inf")

    return {
        "n_trades": int(len(trades)),
        "win_rate": win_rate,
        "avg_return": avg_return,
        "median_return": med_return,
        "sharpe": sharpe,
        "profit_factor": profit_factor,
        "tp_hit_rate": float((trades["exit_reason"] == "TP").mean()),
        "sl_hit_rate": float((trades["exit_reason"] == "SL").mean()),
        "time_rate": float((trades["exit_reason"] == "TIME").mean()),
    }


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Load frozen best config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    filt_name = cfg["filter"]
    tp = float(cfg["tp"])
    sl = float(cfg["sl"])
    hold = int(cfg["hold"])

    print(f"[INFO] Loaded config: filter={filt_name} tp={tp:.0%} sl={sl:.0%} hold={hold}d")

    # Load events
    events = pd.read_csv(args.events)
    for col in ["ticker", "trigger_date"]:
        if col not in events.columns:
            raise ValueError(f"Events file missing required column: {col}")
    if "sector" not in events.columns:
        events["sector"] = "Unknown"

    events["trigger_date"] = normalize_dates(events["trigger_date"])
    events = events[events["trigger_date"].notna()].copy()

    # Optional date window for running
    if args.start:
        start = pd.to_datetime(args.start)
        events = events[events["trigger_date"] >= start]
    if args.end:
        end = pd.to_datetime(args.end)
        events = events[events["trigger_date"] <= end]

    print(f"[INFO] Events in run window: {len(events)}")

    # Load prices
    prices = pd.read_parquet(args.prices)
    prices["Date"] = pd.to_datetime(prices["Date"])
    if getattr(prices["Date"].dt, "tz", None) is not None:
        prices["Date"] = prices["Date"].dt.tz_convert(None)
    prices["Date"] = prices["Date"].dt.normalize()
    prices = prices.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    prices_by_ticker = {t: df.reset_index(drop=True) for t, df in prices.groupby("Ticker")}

    # Apply the same filter definition
    filters = build_filters(events)
    if filt_name not in filters:
        raise ValueError(f"Filter '{filt_name}' not recognized. Available: {list(filters.keys())}")
    ev = filters[filt_name]
    print(f"[INFO] Events after filter '{filt_name}': {len(ev)}")

    # Run trades
    trades = run_backtest(ev, prices_by_ticker, tp_pct=tp, sl_pct=sl, max_hold_days=hold)
    out_trades = os.path.join(args.outdir, "run_trades.csv")
    trades.to_csv(out_trades, index=False)
    print(f"[OK] wrote {out_trades} (n_trades={len(trades)})")

    metrics = analyze_trades(trades)
    out_metrics = os.path.join(args.outdir, "run_metrics.csv")
    pd.DataFrame([metrics if metrics else {}]).to_csv(out_metrics, index=False)
    print(f"[OK] wrote {out_metrics}")

    print("\n=== RUN RESULTS ===")
    if metrics:
        print(metrics)
    else:
        print("No trades generated for this window/filter.")


if __name__ == "__main__":
    main()
