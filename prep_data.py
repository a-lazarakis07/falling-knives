#!/usr/bin/env python3
# prepare_data.py
# Fetch daily OHLCV for many tickers, attach Sector, compute features + drawdowns.

import argparse, os, time
import numpy as np
import pandas as pd
import yfinance as yf

def parse_args():
    p = argparse.ArgumentParser("Fetch OHLCV + Sector, compute features & drawdowns")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--tickers", nargs="+", help="Space-separated tickers")
    g.add_argument("--tickers-file", type=str, help="Text file, one ticker per line")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--outdir", default="data")
    p.add_argument("--sleep", type=float, default=0.2)
    return p.parse_args()

def load_tickers(args):
    if args.tickers:
        return [t.strip().upper() for t in args.tickers if t.strip()]
    with open(args.tickers_file, "r", encoding="utf-8") as f:
        return [ln.strip().upper() for ln in f if ln.strip() and not ln.startswith("#")]

def get_sector_safe(ticker: str) -> str:
    """Try yfinance .get_info; fall back to Unknown."""
    try:
        info = yf.Ticker(ticker).get_info()
        sec = info.get("sector")
        if sec and isinstance(sec, str):
            return sec
    except Exception:
        pass
    return "Unknown"

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    tickers = load_tickers(args)
    rows = []
    for i, t in enumerate(tickers, 1):
        try:
            sec = get_sector_safe(t)
            h = yf.Ticker(t).history(start=args.start, end=args.end, interval="1d", auto_adjust=False)
            if h.empty:
                print(f"[{i}/{len(tickers)}] {t}: no data")
                continue
            h = h.reset_index()
            h["Ticker"] = t
            h["Sector"] = sec
            rows.append(h[["Date","Ticker","Sector","Open","High","Low","Close","Volume"]])
            print(f"[{i}/{len(tickers)}] {t}: ok (Sector={sec})")
        except Exception as e:
            print(f"[{i}/{len(tickers)}] {t}: ERROR -> {e}")
        time.sleep(args.sleep)

    if not rows:
        raise RuntimeError("No data fetched.")
    prices = pd.concat(rows, ignore_index=True)

    # ---------- features (Steps 3–4) ----------
    prices["Date"] = pd.to_datetime(prices["Date"])
    if getattr(prices["Date"].dt, "tz", None) is not None:
        prices["Date"] = prices["Date"].dt.tz_convert(None)

    prices = prices.sort_values(["Ticker","Date"]).reset_index(drop=True)

    # Returns & MAs
    prices["Return"] = prices.groupby("Ticker")["Close"].pct_change()
    prices["MA20"]   = prices.groupby("Ticker")["Close"].transform(lambda s: s.rolling(20).mean())
    prices["MA50"]   = prices.groupby("Ticker")["Close"].transform(lambda s: s.rolling(50).mean())

    # Rolling 3y high & drawdown (need min 252 trading days)
    prices["ROLL_MAX_3Y"] = prices.groupby("Ticker")["Close"].transform(
        lambda s: s.rolling(252*3, min_periods=252).max()
    )
    prices["DD_3Y"] = (prices["Close"] / prices["ROLL_MAX_3Y"]) - 1.0

    # Falling-knife trigger flag (≥30% below rolling 3y high)
    prices["KnifeTrigger"] = prices["DD_3Y"] <= -0.30

    # Save outputs
    raw_path = os.path.join(args.outdir, "prices.parquet")
    dd_path  = os.path.join(args.outdir, "prices_dd.parquet")
    prices.to_parquet(dd_path, index=False)
    prices.drop(columns=["ROLL_MAX_3Y","DD_3Y","KnifeTrigger","MA20","MA50","Return"]).to_parquet(raw_path, index=False)

    print(f"[OK] saved {dd_path}  (with Sector, MAs, DD_3Y, KnifeTrigger)")
    print(f"[OK] saved {raw_path}  (raw OHLCV + Sector)")

if __name__ == "__main__":
    main()