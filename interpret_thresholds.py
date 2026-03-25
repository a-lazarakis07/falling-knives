#!/usr/bin/env python3
"""
interpret_thresholds.py

Interpret the results of threshold_sweep.py and select
a realistic, statistically robust "best threshold" per sector.

Compares thresholds using a balanced score:
    score = 0.6 * recovery_rate
            - 0.2 * normalized_recovery_time
            + 0.2 * normalized_n_events

Prevents unrealistic threshold choices like 40% (too few events).
Creates optional plots of threshold vs recovery rate for each sector.

Run:
    python interpret_thresholds.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- CONFIG ---------------- #

SUMMARY_PATH = "data/threshold_sector_summary_train.csv"
OUTDIR = "data"


# Minimum number of events required for a threshold to be “trustworthy”
MIN_EVENTS_PER_POINT = 150  

# Make per-sector PNG plots?
MAKE_PLOTS = True

# ---------------------------------------- #


def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'score' column combining:
    - recovery_rate (positive)
    - median_days_to_recovery (negative)
    - n_events (positive)
    """

    df = df.copy()

    # Normalize quantities within each sector so scoring is fair
    df["recovery_rate_norm"] = df["recovery_rate"]
    df["n_events_norm"] = df["n_events"] / df["n_events"].max()

    # Handle recovery time normalization (lower is better)
    df["recovery_time_norm"] = df["median_days_to_recovery"] / df["median_days_to_recovery"].max()

    # Balanced scoring system
    df["score"] = (
        0.6 * df["recovery_rate_norm"]
        - 0.2 * df["recovery_time_norm"]
        + 0.2 * df["n_events_norm"]
    )

    return df


def choose_best_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each sector, pick the threshold with the highest score.
    Only consider thresholds with at least MIN_EVENTS_PER_POINT events.
    If none meet that criterion, relax it (fallback).
    """

    best_rows = []
    sectors = df["sector"].unique()

    for sector in sectors:
        g = df[df["sector"] == sector].copy()

        # Strong filtering: need enough events to avoid unreliable deep crashes
        g_valid = g[g["n_events"] >= MIN_EVENTS_PER_POINT].copy()

        # If nothing survives the filter → fallback to all data
        if g_valid.empty:
            g_valid = g

        # Choose best by score
        g_valid = compute_scores(g_valid)
        best = g_valid.sort_values("score", ascending=False).iloc[0]
        best_rows.append(best)

    return pd.DataFrame(best_rows)


def print_summary_table(best_df: pd.DataFrame):
    """
    Pretty-print summary table.
    """
    print("\n=== Best Threshold per Sector ===")

    df = best_df.copy()

    df["dd_threshold"] = (df["dd_threshold"] * 100).round(1)
    df["recovery_rate"] = (df["recovery_rate"] * 100).round(1)
    df["avg_drawdown"] = (df["avg_drawdown"] * 100).round(1)

    cols = [
        "sector",
        "dd_threshold",
        "recovery_rate",
        "median_days_to_recovery",
        "n_events",
        "n_tickers",
        "avg_drawdown",
        "score"
    ]

    print(df[cols].sort_values("sector").to_string(index=False))


def make_plots(df: pd.DataFrame):
    """
    Plot recovery_rate vs threshold for each sector.
    """
    os.makedirs(OUTDIR, exist_ok=True)
    sectors = sorted(df["sector"].unique())

    for sector in sectors:
        g = df[df["sector"] == sector].sort_values("dd_threshold")

        if g.empty:
            continue

        x = g["dd_threshold"] * 100
        y = g["recovery_rate"] * 100

        plt.figure(figsize=(6,4))
        plt.plot(x, y, marker="o")
        plt.title(f"{sector}: Recovery Rate vs Drawdown Threshold")
        plt.xlabel("Drawdown threshold (%)")
        plt.ylabel("Recovery rate (+20%) (%)")
        plt.grid(alpha=0.3)
        plt.tight_layout()

        fname = f"threshold_curve_{sector.replace(' ', '_')}.png"
        path = os.path.join(OUTDIR, fname)
        plt.savefig(path, dpi=160)
        plt.close()
        print(f"[OK] saved {path}")


def main():
    if not os.path.exists(SUMMARY_PATH):
        raise FileNotFoundError(f"Missing file: {SUMMARY_PATH}. Run threshold_sweep.py first.")

    df = pd.read_csv(SUMMARY_PATH)

    # Clean types just in case
    df["dd_threshold"] = df["dd_threshold"].astype(float)
    df["recovery_rate"] = df["recovery_rate"].astype(float)
    df["median_days_to_recovery"] = df["median_days_to_recovery"].astype(float)
    df["n_events"] = df["n_events"].astype(int)

    # Score + choose best thresholds
    best_df = choose_best_thresholds(df)

    # Print summary
    print_summary_table(best_df)

    # Save results
    outpath = os.path.join(OUTDIR, "best_thresholds_by_sector_train.csv")
    best_df.to_csv(outpath, index=False)
    print(f"\n[OK] Saved best thresholds to {outpath}")

    # Make charts
    if MAKE_PLOTS:
        make_plots(df)


if __name__ == "__main__":
    main()
