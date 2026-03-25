#!/usr/bin/env python3
"""
optimize_train_test.py

Train on pre-2022, then test on 2022–2025 (no peeking).

Entry: close on the next available trading day on/after trigger_date (tradable).
Exit: TP/SL/TIME only (no prove-it), same as your optimize_filtered script.

Outputs:
- data/train_optimization_results.csv
- data/best_train_config.json (small json text)
- data/test_trades_best_config.csv
- data/test_metrics_best_config.csv
"""

import os
import json
import time
import argparse
import numpy as np
import pandas as pd


EVENTS_PATH_DEFAULT = "data/event_metrics_test.csv"   # can include all years
PRICES_PATH_DEFAULT = "data/prices_dd.parquet"
OUTDIR_DEFAULT = "data"


def parse_args():
    p = argparse.ArgumentParser("Train on <2022, test on 2022–2025")

    p.add_argument("--events", default=EVENTS_PATH_DEFAULT)
    p.add_argument("--prices", default=PRICES_PATH_DEFAULT)
    p.add_argument("--outdir", default=OUTDIR_DEFAULT)

    # Train/test split
    p.add_argument("--train-end", default="2021-12-31", help="YYYY-MM-DD (inclusive)")
    p.add_argument("--test-start", default="2022-01-01", help="YYYY-MM-DD (inclusive)")
    p.add_argument("--test-end", default="2025-12-31", help="YYYY-MM-DD (inclusive)")

    # Minimum trades on TRAIN to consider config valid
    p.add_argument("--min-trades-train", type=int, default=50)

    # Selection metric on TRAIN
    p.add_argument("--select-metric", default="sharpe", choices=["sharpe", "avg_return", "profit_factor"],
                   help="Metric to pick best config from TRAIN only")

    return p.parse_args()


def normalize_dates(s: pd.Series) -> pd.Series:
    s = pd.to_datetime(s, errors="coerce")
    # drop tz if present, normalize to midnight
    try:
        s = s.dt.tz_localize(None)
    except Exception:
        pass
    return s.dt.normalize()


def build_filters(events: pd.DataFrame) -> dict:
    """Same filter ideas you used, built from the passed-in events slice."""
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


def default_param_grid() -> list[dict]:
    return [
        {"tp": 0.15, "sl": 0.08, "hold": 63},
        {"tp": 0.20, "sl": 0.08, "hold": 90},
        {"tp": 0.20, "sl": 0.10, "hold": 90},
        {"tp": 0.25, "sl": 0.10, "hold": 90},
        {"tp": 0.30, "sl": 0.10, "hold": 120},
        {"tp": 0.30, "sl": 0.12, "hold": 120},
        {"tp": 0.35, "sl": 0.12, "hold": 120},
        {"tp": 0.40, "sl": 0.15, "hold": 120},
    ]


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
            "trade_return": (exit_price / entry_price) - 1.0,
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

    train_end = pd.to_datetime(args.train_end)
    test_start = pd.to_datetime(args.test_start)
    test_end = pd.to_datetime(args.test_end)

    print("[INFO] Loading events...")
    events = pd.read_csv(args.events)

    for col in ["ticker", "trigger_date"]:
        if col not in events.columns:
            raise ValueError(f"Events file missing required column: {col}")

    if "sector" not in events.columns:
        events["sector"] = "Unknown"

    events["trigger_date"] = normalize_dates(events["trigger_date"])
    events = events[events["trigger_date"].notna()].copy()

    # Split
    train_events = events[events["trigger_date"] <= train_end].copy()
    test_events = events[(events["trigger_date"] >= test_start) & (events["trigger_date"] <= test_end)].copy()

    print(f"[INFO] Train: <= {train_end.date()}  (events={len(train_events)})")
    print(f"[INFO] Test:  {test_start.date()} -> {test_end.date()} (events={len(test_events)})")

    if train_events.empty:
        raise ValueError("No TRAIN events found. Your events file may only contain 2022–2025.")
    if test_events.empty:
        raise ValueError("No TEST events found in 2022–2025 in the events file.")

    print("[INFO] Loading prices...")
    prices = pd.read_parquet(args.prices)
    prices["Date"] = pd.to_datetime(prices["Date"])
    if getattr(prices["Date"].dt, "tz", None) is not None:
        prices["Date"] = prices["Date"].dt.tz_convert(None)
    prices["Date"] = prices["Date"].dt.normalize()
    prices = prices.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    prices_by_ticker = {t: df.reset_index(drop=True) for t, df in prices.groupby("Ticker")}

    # TRAIN optimization
    train_filters = build_filters(train_events)
    param_grid = default_param_grid()

    total = len(train_filters) * len(param_grid)
    print(f"\n[INFO] TRAIN optimization: {len(train_filters)} filters × {len(param_grid)} params = {total} configs")
    start_t = time.time()

    train_rows = []
    i = 0
    for fname, fev in train_filters.items():
        if fev.empty:
            continue
        for p in param_grid:
            i += 1
            if i % 20 == 0 or i == 1:
                print(f"[PROGRESS] {i}/{total}  filter={fname}  TP={p['tp']:.0%} SL={p['sl']:.0%} hold={p['hold']}")

            trades = run_backtest(fev, prices_by_ticker, tp_pct=p["tp"], sl_pct=p["sl"], max_hold_days=p["hold"])
            m = analyze_trades(trades)
            if m is None or m["n_trades"] < int(args.min_trades_train):
                continue

            train_rows.append({
                "filter": fname,
                "tp": p["tp"],
                "sl": p["sl"],
                "hold": p["hold"],
                **m
            })

    elapsed = time.time() - start_t
    print(f"[OK] TRAIN optimization done in {elapsed:.1f}s")

    if not train_rows:
        raise ValueError("No valid TRAIN configs met min-trades threshold. Lower --min-trades-train or expand events history.")

    train_df = pd.DataFrame(train_rows)
    out_train = os.path.join(args.outdir, "train_optimization_results.csv")
    train_df.to_csv(out_train, index=False)
    print(f"[OK] wrote {out_train}")

    metric = args.select_metric
    best = train_df.sort_values(metric, ascending=False).iloc[0].to_dict()

    best_cfg = {
        "filter": best["filter"],
        "tp": float(best["tp"]),
        "sl": float(best["sl"]),
        "hold": int(best["hold"]),
        "selected_by": metric,
        "train_metrics": {k: best[k] for k in ["n_trades","win_rate","avg_return","median_return","sharpe","profit_factor","tp_hit_rate","sl_hit_rate","time_rate"]},
        "train_window_end": str(train_end.date()),
        "test_window": [str(test_start.date()), str(test_end.date())],
    }

    out_best = os.path.join(args.outdir, "best_train_config.json")
    with open(out_best, "w", encoding="utf-8") as f:
        json.dump(best_cfg, f, indent=2)
    print(f"[OK] wrote {out_best}")

    print("\n=== BEST CONFIG FROM TRAIN ===")
    print(f"Filter: {best_cfg['filter']} | TP: {best_cfg['tp']:.0%} | SL: {best_cfg['sl']:.0%} | Hold: {best_cfg['hold']}d")
    print("Train metrics:", best_cfg["train_metrics"])

    # TEST evaluation (apply same filter definition, but computed on TEST slice)
    test_filters = build_filters(test_events)
    if best_cfg["filter"] not in test_filters:
        raise ValueError(f"Best filter {best_cfg['filter']} not found in test filters (unexpected).")
    test_ev = test_filters[best_cfg["filter"]]

    test_trades = run_backtest(test_ev, prices_by_ticker, tp_pct=best_cfg["tp"], sl_pct=best_cfg["sl"], max_hold_days=best_cfg["hold"])
    test_metrics = analyze_trades(test_trades)

    out_test_trades = os.path.join(args.outdir, "test_trades_best_config.csv")
    test_trades.to_csv(out_test_trades, index=False)
    print(f"\n[OK] wrote {out_test_trades} (n_trades={len(test_trades)})")

    out_test_metrics = os.path.join(args.outdir, "test_metrics_best_config.csv")
    pd.DataFrame([test_metrics if test_metrics else {}]).to_csv(out_test_metrics, index=False)
    print(f"[OK] wrote {out_test_metrics}")

    print("\n=== TEST RESULTS (2022–2025) USING TRAINED CONFIG ===")
    if not test_metrics:
        print("No test trades generated.")
    else:
        print(test_metrics)


if __name__ == "__main__":
    main()