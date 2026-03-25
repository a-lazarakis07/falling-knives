#!/usr/bin/env python3
# summarize_by_sector.py
# Build per-ticker and per-sector summaries; simple sector charts.

import argparse, os
import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser("Summarize events by ticker and sector")
    p.add_argument("--events", default="data/event_metrics.csv")
    p.add_argument("--outdir", default="data")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    events = pd.read_csv(
        args.events,
        parse_dates=["trigger_date","bottom_date","recovery_date","ma50_cross_date"]
    )
    rec_col = [c for c in events.columns if c.startswith("recovered_+")][0]

    # -------- per-ticker summary --------
    per_ticker = (events
        .groupby(["sector","ticker"])
        .agg(
            n_events=("ticker","count"),
            recovery_rate=(rec_col,"mean"),
            median_days_to_bottom=("days_to_bottom","median"),
            median_days_to_recovery=("days_bottom_to_recovery","median"),
            median_days_to_ma50=("days_bottom_to_ma50","median"),
            avg_trigger_dd=("drawdown_at_trigger","mean"),
        )
        .reset_index()
    )
    per_ticker_path = os.path.join(args.outdir, "ticker_summary.csv")
    per_ticker.to_csv(per_ticker_path, index=False)
    print(f"[OK] wrote {per_ticker_path}")

    # -------- per-sector rollup --------
    per_sector = (per_ticker
        .groupby("sector")
        .agg(
            avg_recovery_rate=("recovery_rate","mean"),
            avg_days_to_recovery=("median_days_to_recovery","mean"),
            avg_drawdown=("avg_trigger_dd","mean"),
            n_tickers=("ticker","count"),
            total_events=("n_events","sum"),
        )
        .reset_index()
        .sort_values("avg_recovery_rate", ascending=False)
    )
    per_sector_path = os.path.join(args.outdir, "sector_summary.csv")
    per_sector.to_csv(per_sector_path, index=False)
    print(f"[OK] wrote {per_sector_path}")
    print(per_sector)

    # -------- quick bar charts --------
    plt.figure(figsize=(9,4))
    plt.bar(per_sector["sector"], per_sector["avg_recovery_rate"])
    plt.title("Average recovery rate (+X%) by sector")
    plt.ylabel("Fraction of events recovered")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "sector_recovery_rate.png"), dpi=160)

    plt.figure(figsize=(9,4))
    plt.bar(per_sector["sector"], per_sector["avg_days_to_recovery"])
    plt.title("Average days to recovery by sector")
    plt.ylabel("Days")
    plt.xticks(rotation=45, ha="right"); plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "sector_days_to_recovery.png"), dpi=160)

if __name__ == "__main__":
    main()
