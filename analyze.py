#!/usr/bin/env python3
# Find drawdown events per ticker (trigger -> bottom -> recovery), using
# sector-specific drawdown thresholds, and keep Sector info.

import argparse
import os
import pandas as pd

# Default global constants (used as fallback)
DD_THRESHOLD = 0.30         # fallback if sector not in mapping
DEBOUNCE_DAYS = 63
BOTTOM_WINDOW_DAYS = 60
RECOVERY_GAIN = 0.20
RECOVERY_WINDOW = 252       # days after bottom (calendar window)

BEST_THRESHOLDS_PATH = "data/best_thresholds_by_sector_train.csv"


def parse_args():

    p = argparse.ArgumentParser("Analyze falling-knife events with sector-specific thresholds")
    p.add_argument("--prices-dd", default="data/prices_dd.parquet",
                   help="Path to prepared price data with DD_3Y, MA50, Sector")
    p.add_argument("--outdir", default="data")
    p.add_argument("--dd-threshold", type=float, default=DD_THRESHOLD,
                   help="Fallback global drawdown threshold (if sector not found)")
    p.add_argument("--best-thresholds", default=BEST_THRESHOLDS_PATH,
                   help="CSV with best dd_threshold per sector")
    p.add_argument("--debounce-days", type=int, default=DEBOUNCE_DAYS)
    p.add_argument("--bottom-window-days", type=int, default=BOTTOM_WINDOW_DAYS)
    p.add_argument("--recovery-gain", type=float, default=RECOVERY_GAIN)
    p.add_argument("--recovery-window-days", type=int, default=RECOVERY_WINDOW)
    p.add_argument("--test-start", default="2022-01-01",
               help="Only keep events whose trigger_date is on/after this date (YYYY-MM-DD)")
    p.add_argument("--test-end", default="2025-12-31",
               help="Only keep events whose trigger_date is on/before this date (YYYY-MM-DD)")
    return p.parse_args()


def load_sector_thresholds(path: str) -> dict:
    """
    Load sector-specific drawdown thresholds from CSV.
    Expected columns: sector, dd_threshold
    Returns dict: { sector_name: dd_threshold_float }
    """
    if not os.path.exists(path):
        print(f"[WARN] best thresholds file not found: {path} -> using global dd_threshold only")
        return {}

    df = pd.read_csv(path)
    if "sector" not in df.columns or "dd_threshold" not in df.columns:
        print(f"[WARN] best thresholds file at {path} missing required columns -> using global dd_threshold only")
        return {}

    mapping = {
        row["sector"]: float(row["dd_threshold"])
        for _, row in df.iterrows()
    }
    print("[INFO] loaded sector-specific thresholds:")
    for s, th in mapping.items():
        print(f"  {s}: {th:.2%}")
    return mapping


def get_triggers(df_t: pd.DataFrame,
                 dd_threshold: float,
                 min_gap_days: int) -> list:
    """
    For one ticker's DataFrame (single sector), find trigger dates where
    DD_3Y <= -dd_threshold, with debounce spacing.
    """
    # DD_3Y is negative for drawdowns, so we compare to -dd_threshold
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
                        rec_window_days: int,
                        dd_threshold_used: float):
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

    # 1) bottom in next N trading rows
    fw = df_t.iloc[trig_idx: trig_idx + bottom_window_days + 1]
    if fw.empty:
        return None

    i_min = fw["Close"].idxmin()
    bottom_row = df_t.loc[i_min]
    bottom_date = bottom_row["Date"]
    bottom_close = float(bottom_row["Close"])
    days_to_bottom = int((bottom_date - trigger_date).days)

    # 2) recovery +X% within window (by calendar days)
    recovery_level = bottom_close * (1 + rec_gain)
    future = df_t[
        (df_t["Date"] >= bottom_date) &
        (df_t["Date"] <= bottom_date + pd.Timedelta(days=rec_window_days))
    ]
    hit = future[future["Close"] >= recovery_level]
    if not hit.empty:
        recovered = 1
        rec_date = hit.iloc[0]["Date"]
        days_bottom_to_recovery = int((rec_date - bottom_date).days)
    else:
        recovered = 0
        rec_date = None
        days_bottom_to_recovery = None

    # 3) mean-reversion proxy: first Close >= MA50
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
        "dd_threshold_used": dd_threshold_used,  # <— store which threshold was used
        "trigger_date": trigger_date,
        "trigger_close": float(row_trig["Close"]),
        "drawdown_at_trigger": float(row_trig["DD_3Y"]),
        "bottom_date": bottom_date,
        "bottom_close": bottom_close,
        "days_to_bottom": days_to_bottom,
        f"recovered_+{int(rec_gain*100)}%": recovered,
        "recovery_date": rec_date,
        "days_bottom_to_recovery": days_bottom_to_recovery,
        "crossed_ma50": crossed_ma50,
        "ma50_cross_date": ma50_date,
        "days_bottom_to_ma50": days_bottom_to_ma50,
    }


def main():
    args = parse_args()
    test_start = pd.to_datetime(args.test_start)
    test_end   = pd.to_datetime(args.test_end)

    os.makedirs(args.outdir, exist_ok=True)

    # Load sector-specific thresholds (may be empty if file missing)
    sector_thresholds = load_sector_thresholds(args.best_thresholds)

    # Load price data
    prices = pd.read_parquet(args.prices_dd)
    prices["Date"] = pd.to_datetime(prices["Date"])
    if getattr(prices["Date"].dt, "tz", None) is not None:
        prices["Date"] = prices["Date"].dt.tz_convert(None)

    prices = prices.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    events_rows = []

    for tkr, df_t in prices.groupby("Ticker"):
        df_t = df_t.reset_index(drop=True)
        sector = df_t["Sector"].iloc[0]

        # Decide which threshold to use for this ticker
        dd_th = sector_thresholds.get(sector, args.dd_threshold)

        # Find triggers using sector-specific threshold
        triggers = get_triggers(df_t, dd_th, args.debounce_days)
        if not triggers:
            continue
        # Keep only triggers in the TEST window
        triggers = [d for d in triggers if (d >= test_start and d <= test_end)]
        if not triggers:
           continue


        for td in triggers:
            res = analyze_one_trigger(
                df_t,
                td,
                bottom_window_days=args.bottom_window_days,
                rec_gain=args.recovery_gain,
                rec_window_days=args.recovery_window_days,
                dd_threshold_used=dd_th,
            )
            if res:
                events_rows.append(res)

    events = pd.DataFrame(events_rows)
    out_path = os.path.join(args.outdir, "event_metrics_test.csv")
    events.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path} (rows={len(events)})")


if __name__ == "__main__":
    main()
