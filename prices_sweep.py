#!/usr/bin/env python3
"""

Sweep over different drawdown thresholds (e.g., 20% to 40% in 1% steps),
re-run the falling-knife event logic for each threshold, and summarize
recovery behavior by sector.

Inputs:
  - data/prices_dd.parquet  (from your existing pipeline)

Outputs:
  - data/threshold_events.csv
  - data/threshold_sector_summary.csv
"""

import os
import numpy as np
import pandas as pd
import argparse


PRICES_DD_PATH = "data/prices_dd.parquet"
OUTDIR = "data"

MIN_DD = 0.20      # 20% drawdown
MAX_DD = 0.40      # 40% drawdown
STEP_DD = 0.01     # 1% increments

DEBOUNCE_DAYS = 63         # spacing between triggers
BOTTOM_WINDOW_DAYS = 60    # days after trigger to search for bottom
RECOVERY_GAIN = 0.20       # +20% from bottom
RECOVERY_WINDOW = 252      # days after bottom to search for recovery


# --------------------------------------------------------------- # 
def parse_args():
    p = argparse.ArgumentParser("Sweep thresholds on TRAIN data only")
    p.add_argument("--prices-dd", default=PRICES_DD_PATH)
    p.add_argument("--outdir", default=OUTDIR)
    p.add_argument("--train-end", default="2021-12-31",
                   help="Last date included in TRAIN set (YYYY-MM-DD).")
    return p.parse_args()

def get_triggers_for_threshold(df_t: pd.DataFrame,
                               dd_threshold: float,
                               min_gap_days: int) -> list[pd.Timestamp]:
    """
    For one ticker df_t, find trigger dates where drawdown <= -dd_threshold.
    Debounce so we don't get multiple triggers for the same crash.
    """
    # DD_3Y is negative for drawdowns; we use <= -threshold
    mask = df_t["DD_3Y"] <= -dd_threshold
    candidates = df_t.loc[mask, "Date"]

    triggers = []
    last = None
    for d in candidates:
        if last is None or (d - last).days >= min_gap_days:
            triggers.append(d)
            last = d
    return triggers


def analyze_one_trigger(df_t: pd.DataFrame,
                        trigger_date: pd.Timestamp,
                        bottom_window_days: int,
                        rec_gain: float,
                        rec_window_days: int):
    """
    Given a ticker's time series and a trigger_date, compute:
      - bottom within bottom_window_days
      - whether price recovers +rec_gain from bottom within rec_window_days
      - when price crosses MA50 after bottom
    Returns a dict with event metrics, or None if not enough data.
    """
    row_trig = df_t.loc[df_t["Date"] == trigger_date]
    if row_trig.empty:
        return None
    row_trig = row_trig.iloc[0]
    trig_idx = row_trig.name

    # 1) find bottom in next N trading bars
    fw = df_t.iloc[trig_idx : trig_idx + bottom_window_days + 1]
    if fw.empty:
        return None

    i_min = fw["Close"].idxmin()
    bottom_row = df_t.loc[i_min]
    bottom_date = bottom_row["Date"]
    bottom_close = float(bottom_row["Close"])
    days_to_bottom = int((bottom_date - trigger_date).days)

    # 2) recovery: +X% from bottom within rec_window_days (calendar)
    recovery_level = bottom_close * (1 + rec_gain)
    future = df_t[(df_t["Date"] >= bottom_date) &
                  (df_t["Date"] <= bottom_date + pd.Timedelta(days=rec_window_days))]

    hit = future[future["Close"] >= recovery_level]
    if not hit.empty:
        recovered = 1
        rec_date = hit.iloc[0]["Date"]
        days_bottom_to_recovery = int((rec_date - bottom_date).days)
    else:
        recovered = 0
        rec_date = None
        days_bottom_to_recovery = None

    # 3) mean-reversion proxy: first Close >= MA50 after bottom
    cross = future[future["Close"] >= future["MA50"]]
    if not cross.empty:
        ma50_date = cross.iloc[0]["Date"]
        days_bottom_to_ma50 = int((ma50_date - bottom_date).days)
        crossed_ma50 = 1
    else:
        ma50_date = None
        days_bottom_to_ma50 = None
        crossed_ma50 = 0

    return {
        "ticker": df_t["Ticker"].iloc[0],
        "sector": df_t["Sector"].iloc[0],
        "trigger_date": trigger_date,
        "trigger_close": float(row_trig["Close"]),
        "drawdown_at_trigger": float(row_trig["DD_3Y"]),
        "bottom_date": bottom_date,
        "bottom_close": bottom_close,
        "days_to_bottom": days_to_bottom,
        "recovered_+20%": recovered,
        "recovery_date": rec_date,
        "days_bottom_to_recovery": days_bottom_to_recovery,
        "crossed_ma50": crossed_ma50,
        "ma50_cross_date": ma50_date,
        "days_bottom_to_ma50": days_bottom_to_ma50,
    }


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    prices = pd.read_parquet(args.prices_dd)
    prices["Date"] = pd.to_datetime(prices["Date"])

# --- FIX TZ-AWARE VS TZ-NAIVE COMPARISON ---
# If Date is tz-aware (e.g., America/New_York), drop tz info so comparisons work
    if getattr(prices["Date"].dt, "tz", None) is not None:
        prices["Date"] = prices["Date"].dt.tz_convert(None)

    prices = prices.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    train_end = pd.to_datetime(args.train_end)
    prices = prices[prices["Date"] <= train_end].copy()



    # 2) generate thresholds: e.g. 0.20, 0.21, ..., 0.40
    thresholds = np.arange(MIN_DD, MAX_DD + 1e-9, STEP_DD)
    print("Running thresholds:", [round(t, 3) for t in thresholds])

    all_events = []

    # 3) loop over thresholds
    for th in thresholds:
        print(f"\n=== Threshold {th:.2%} drawdown ===")
        for tkr, df_t in prices.groupby("Ticker"):
            df_t = df_t.reset_index(drop=True)
            triggers = get_triggers_for_threshold(df_t, dd_threshold=th, min_gap_days=DEBOUNCE_DAYS)
            if not triggers:
                continue
            for td in triggers:
                res = analyze_one_trigger(df_t, td,
                                          bottom_window_days=BOTTOM_WINDOW_DAYS,
                                          rec_gain=RECOVERY_GAIN,
                                          rec_window_days=RECOVERY_WINDOW)
                if res is not None:
                    res["dd_threshold"] = th  # store which threshold produced this event
                    all_events.append(res)

    events = pd.DataFrame(all_events)
    events_path = os.path.join(args.outdir, "threshold_events_train.csv")

    events.to_csv(events_path, index=False)
    print(f"[OK] wrote {events_path} (rows={len(events)})")

    # 4) summarize by (threshold, sector)
    if events.empty:
        print("No events found – check thresholds or data.")
        return

    sector_summary = (
        events
        .groupby(["dd_threshold", "sector"])
        .agg(
            n_events=("ticker", "count"),
            recovery_rate=("recovered_+20%", "mean"),
            median_days_to_recovery=("days_bottom_to_recovery", "median"),
            avg_drawdown=("drawdown_at_trigger", "mean"),
            n_tickers=("ticker", "nunique"),
        )
        .reset_index()
        .sort_values(["dd_threshold", "sector"])
    )

    summary_path = os.path.join(args.outdir, "threshold_sector_summary_train.csv")
    sector_summary.to_csv(summary_path, index=False)
    print(f"[OK] wrote {summary_path}")


if __name__ == "__main__":
    main()
