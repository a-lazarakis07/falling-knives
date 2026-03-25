#!/usr/bin/env python3
"""
sector_optimize.py

Optimize exit parameters for ONE sector using your bottom-entry backtest logic.
This tells us:
- whether a sector really has stable rebound edge
- what holding period / TP / SL / prove rules maximize Sharpe-like while keeping win rate high

Inputs:
- events CSV: must include ticker, sector, bottom_date, bottom_close (or any of bottom_price candidates)
- prices parquet: prices_dd.parquet must include Date, Ticker, Close

Outputs:
- data/sector_opt_results_<SECTOR>.csv (all tried parameter combos)
- prints top 20 configs by Sharpe-like with win rate constraint
"""

import os
import argparse
import numpy as np
import pandas as pd


BOTTOM_PRICE_CANDIDATES = ["bottom_close", "bottom_price", "bottom_close_price", "bottom_px"]


def parse_args():
    p = argparse.ArgumentParser("Optimize exit rules for a single sector (bottom-entry upper bound)")
    p.add_argument("--events", default="data/event_metrics_test.csv")
    p.add_argument("--prices", default="data/prices_dd.parquet")
    p.add_argument("--outdir", default="data")
    p.add_argument("--sector", default="Energy")

    # Constraints / objective filters
    p.add_argument("--min-win-rate", type=float, default=0.90, help="Minimum win rate to consider (0-1)")

    return p.parse_args()


def to_dt(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")


def pick_bottom_price_col(events: pd.DataFrame) -> str:
    for c in BOTTOM_PRICE_CANDIDATES:
        if c in events.columns:
            return c
    raise ValueError(f"Could not find bottom price column; expected one of {BOTTOM_PRICE_CANDIDATES}")


def run_exit(fut: pd.DataFrame, entry_date: pd.Timestamp, entry_price: float,
             tp_pct: float, sl_pct: float, prove_days: int, prove_pct: float, max_hold_days: int):
    tp = entry_price * (1.0 + tp_pct)
    sl = entry_price * (1.0 - sl_pct)

    prove_date = entry_date + pd.Timedelta(days=int(prove_days))
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
            if prove_ret < prove_pct:
                exit_reason = "PROVE_FAIL"
                exit_date = prove_row.Date
                exit_price = float(prove_row.Close)
                break

    trade_return = (exit_price / entry_price) - 1.0
    return trade_return, exit_reason


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    events = pd.read_csv(args.events)
    for c in ["bottom_date", "trigger_date"]:
        to_dt(events, c)

    # Filter to chosen sector
    if "sector" not in events.columns:
        raise ValueError("events file missing column: sector")
    events = events[events["sector"] == args.sector].copy()

    if events.empty:
        raise ValueError(f"No events found for sector={args.sector}. Check spelling vs your CSV.")

    bottom_price_col = pick_bottom_price_col(events)
    print(f"[INFO] sector={args.sector} | using bottom price col: {bottom_price_col} | events={len(events)}")

    prices = pd.read_parquet(args.prices)
    prices["Date"] = pd.to_datetime(prices["Date"])
    if getattr(prices["Date"].dt, "tz", None) is not None:
        prices["Date"] = prices["Date"].dt.tz_convert(None)
    prices = prices.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    prices_by_ticker = {t: df.reset_index(drop=True) for t, df in prices.groupby("Ticker")}

    # Parameter grid (keep it modest to start; you can widen later)
    TP_LIST = [0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
    SL_LIST = [0.06, 0.08, 0.10, 0.12]
    PROVE_DAYS_LIST = [5, 7, 10, 14]
    PROVE_PCT_LIST = [0.00, 0.01, 0.02, 0.03]
    MAX_HOLD_LIST = [21, 42, 63, 84]

    rows = []

    for tp in TP_LIST:
        for sl in SL_LIST:
            for pdays in PROVE_DAYS_LIST:
                for ppct in PROVE_PCT_LIST:
                    for maxhold in MAX_HOLD_LIST:

                        rets = []
                        reasons = {"TP": 0, "SL": 0, "PROVE_FAIL": 0, "TIME": 0}

                        for _, ev in events.iterrows():
                            tkr = ev["ticker"]
                            bd = ev["bottom_date"]
                            bp = ev[bottom_price_col]

                            if pd.isna(bd) or pd.isna(bp):
                                continue

                            df_t = prices_by_ticker.get(tkr)
                            if df_t is None or df_t.empty:
                                continue

                            entry_date = pd.to_datetime(bd)
                            entry_price = float(bp)

                            deadline = entry_date + pd.Timedelta(days=int(maxhold))
                            fut = df_t[(df_t["Date"] >= entry_date) & (df_t["Date"] <= deadline)].copy()
                            if fut.empty:
                                continue

                            r, reason = run_exit(fut, entry_date, entry_price, tp, sl, pdays, ppct, maxhold)
                            rets.append(r)
                            reasons[reason] = reasons.get(reason, 0) + 1

                        if len(rets) < 50:
                            continue  # too few trades -> unstable

                        rets = np.array(rets, dtype=float)
                        win = float((rets > 0).mean())
                        avg = float(rets.mean())
                        med = float(np.median(rets))
                        sd = float(rets.std(ddof=0))
                        sharpe_like = float(avg / sd) if sd > 0 else np.nan

                        rows.append({
                            "sector": args.sector,
                            "n_trades": len(rets),
                            "tp_pct": tp,
                            "sl_pct": sl,
                            "prove_days": pdays,
                            "prove_pct": ppct,
                            "max_hold_days": maxhold,
                            "win_rate": win,
                            "avg_return": avg,
                            "median_return": med,
                            "sharpe_like": sharpe_like,
                            "tp_rate": reasons["TP"] / len(rets),
                            "sl_rate": reasons["SL"] / len(rets),
                            "prove_fail_rate": reasons["PROVE_FAIL"] / len(rets),
                            "time_rate": reasons["TIME"] / len(rets),
                        })

    res = pd.DataFrame(rows)
    out = os.path.join(args.outdir, f"sector_opt_results_{args.sector.replace(' ','_')}.csv")
    res.to_csv(out, index=False)
    print(f"[OK] wrote {out} (rows={len(res)})")

    # Show best configs that satisfy win constraint
    good = res[res["win_rate"] >= float(args.min_win_rate)].copy()
    if good.empty:
        print(f"\n[WARN] No configs met min win rate {args.min_win_rate:.2f}. Try lowering --min-win-rate.")
        top = res.sort_values("sharpe_like", ascending=False).head(20)
    else:
        top = good.sort_values("sharpe_like", ascending=False).head(20)

    pd.set_option("display.max_columns", 200)
    print("\n=== Top configs ===")
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
