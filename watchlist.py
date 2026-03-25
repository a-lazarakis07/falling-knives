#!/usr/bin/env python3
"""
watchlist.py

Scan a universe of tickers (e.g. S&P 500) and flag names that are currently
in "falling knife" territory based on sector-specific drawdown thresholds.

Logic:
- For each ticker:
    - Fetch last ~3 years of daily prices
    - Compute 3-year high (max close)
    - Compute current drawdown = current_close / three_year_high - 1
    - Get sector from yfinance
    - Look up sector-specific threshold from best_thresholds_by_sector.csv
    - If drawdown <= -sector_threshold -> add to watchlist

Outputs:
- data/watchlist_signals.csv
"""

import os
import time
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf


TICKERS_FILE_DEFAULT = "data/sp500_tickers.txt"
BEST_THRESHOLDS_PATH_DEFAULT = "data/best_thresholds_by_sector.csv"
OUTDIR_DEFAULT = "data"

# How many calendar years of history to look back for the high
YEARS_LOOKBACK = 3

# Sleep between API calls (seconds) to be nice to Yahoo
SLEEP_BETWEEN_CALLS = 0.3


def parse_args():
    p = argparse.ArgumentParser("Build falling-knife watchlist using sector-specific thresholds")
    p.add_argument("--tickers-file", default=TICKERS_FILE_DEFAULT,
                   help="Text file with one ticker per line")
    p.add_argument("--best-thresholds", default=BEST_THRESHOLDS_PATH_DEFAULT,
                   help="CSV with best dd_threshold per sector")
    p.add_argument("--outdir", default=OUTDIR_DEFAULT)
    return p.parse_args()


def load_tickers(path: str) -> list[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Tickers file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        tickers = [ln.strip().upper() for ln in f if ln.strip() and not ln.startswith("#")]
    return tickers


def load_sector_thresholds(path: str) -> dict:
    """
    Load sector-specific drawdown thresholds from CSV:
        columns: sector, dd_threshold
    Returns dict { sector: dd_threshold_float }
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Best thresholds file not found: {path}")
    df = pd.read_csv(path)
    if "sector" not in df.columns or "dd_threshold" not in df.columns:
        raise ValueError(f"{path} must contain 'sector' and 'dd_threshold' columns")
    mapping = {row["sector"]: float(row["dd_threshold"]) for _, row in df.iterrows()}
    return mapping


def get_sector_safe(ticker: str) -> str:
    """
    Try to get sector from yfinance. Fallback: 'Unknown'.
    """
    try:
        info = yf.Ticker(ticker).get_info()
        sec = info.get("sector")
        if isinstance(sec, str) and sec.strip():
            return sec.strip()
    except Exception:
        pass
    return "Unknown"


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    tickers = load_tickers(args.tickers_file)
    sector_thresholds = load_sector_thresholds(args.best_thresholds)

    print(f"[INFO] Loaded {len(tickers)} tickers from {args.tickers_file}")
    print("[INFO] Sector thresholds:")
    for s, th in sector_thresholds.items():
        print(f"  {s}: {th:.1%}")

    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=int(365 * YEARS_LOOKBACK))

    watchlist_rows = []

    for i, t in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] Checking {t} ...", end="", flush=True)
        try:
            # Get history
            hist = yf.Ticker(t).history(start=start_date, end=end_date, interval="1d", auto_adjust=False)
            if hist.empty:
                print(" no data")
                time.sleep(SLEEP_BETWEEN_CALLS)
                continue

            hist = hist.reset_index()
            hist = hist.sort_values("Date")
            closes = hist["Close"].astype(float)

            three_year_high = float(closes.max())
            current_price = float(closes.iloc[-1])

            if three_year_high <= 0 or np.isnan(three_year_high):
                print(" invalid high")
                time.sleep(SLEEP_BETWEEN_CALLS)
                continue

            drawdown = current_price / three_year_high - 1.0

            # Sector & threshold
            sector = get_sector_safe(t)
            dd_th = sector_thresholds.get(sector)

            if dd_th is None:
                print(f" sector={sector} (no threshold) -> skip")
                time.sleep(SLEEP_BETWEEN_CALLS)
                continue

            # Check if in falling-knife zone
            in_zone = drawdown <= -dd_th

            if in_zone:
                print(f" HIT (sector={sector}, dd={drawdown:.1%}, th={-dd_th:.1%} from high)")
                watchlist_rows.append({
                    "date": end_date,
                    "ticker": t,
                    "sector": sector,
                    "current_price": current_price,
                    "three_year_high": three_year_high,
                    "drawdown": drawdown,
                    "sector_dd_threshold": dd_th,
                })
            else:
                print(f" ok (sector={sector}, dd={drawdown:.1%})")

        except Exception as e:
            print(f" ERROR: {e}")

        time.sleep(SLEEP_BETWEEN_CALLS)

    if not watchlist_rows:
        print("\n[INFO] No tickers currently in falling-knife zone based on sector thresholds.")
        return

    watchlist = pd.DataFrame(watchlist_rows)
    out_path = os.path.join(args.outdir, "watchlist_signals.csv")
    watchlist.to_csv(out_path, index=False)
    print(f"\n[OK] wrote watchlist -> {out_path} (n_signals={len(watchlist)})")

    # Also show a quick view sorted by deepest drawdown
    print("\n=== Current Falling-Knife Watchlist (sorted by drawdown) ===")
    df_show = watchlist.copy()
    df_show["drawdown"] = (df_show["drawdown"] * 100).round(1)
    df_show["sector_dd_threshold"] = (df_show["sector_dd_threshold"] * 100).round(1)
    print(
        df_show.sort_values("drawdown")[  # most negative first
            ["ticker", "sector", "drawdown", "sector_dd_threshold", "current_price"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
