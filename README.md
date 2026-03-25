# Catching Falling Knives: Mean Reversion in Extreme Equity Drawdowns

**Author:** Angela Lazarakis  
**Institution:** St. George Capital, 2026  
**Paper:** *Catching Falling Knives: A Systematic Study of Mean Reversion in Extreme Equity Drawdowns*

---

## Overview

This repository contains all code used to produce the results in the accompanying research paper. The project develops a rule-based, sector-aware strategy for identifying and trading extreme drawdown events in S&P 500 large-cap equities. Sector-specific drawdown thresholds are learned on pre-2022 data and held fixed for a fully out-of-sample evaluation over 2022–2025.

**Key out-of-sample results:**
- 81.6% win rate across 114 trades
- Sharpe-like ratio of 0.845
- Profit factor of 7.91
- Annualized portfolio return of ~16.3% (equal-weight, mark-to-market)
- Market-adjusted excess return t-statistic of 4.67 (p < 0.0001)

---

## Files

| File | Description |
|---|---|
| `sp500.csv` | Static S&P 500 constituent snapshot |
| `import.py` | Step 1: Parse tickers from sp500.csv |
| `prep_data.py` | Step 2: Fetch OHLCV via yfinance, compute features (DD_3Y, MA50) |
| `prices_sweep.py` | Step 3: Sweep drawdown thresholds on training data |
| `interpret_thresholds.py` | Step 4: Score and select best threshold per sector |
| `analyze.py` | Step 5: Generate falling-knife events |
| `optimize_strategy.py` | Step 6: Train/test split and parameter optimization |
| `backtest_confirm.py` | Step 7: Run frozen strategy on out-of-sample window |
| `results.py` | Step 8: Sector summaries and charts |
| `graph.py` | Price and event visualizations (Figures 1-4) |
| `backtest.py` | Bottom-entry backtest for signal quality upper bound |
| `discover_bottom_patterns.py` | Tradable bottom-confirmation entry exploration |
| `sector_optimize.py` | Per-sector exit parameter optimization |
| `watchlist.py` | Live scanner for current falling-knife signals |

---

## Pipeline: How to Run

### Prerequisites

```bash
pip install yfinance pandas numpy pyarrow matplotlib
```

### Step-by-step

**1. Load tickers**
```bash
python import.py
# Reads sp500.csv → writes data/sp500_tickers.txt
```

**2. Fetch price data**
```bash
python prep_data.py --tickers-file data/sp500_tickers.txt --start 2015-01-01
# Fetches daily OHLCV from yfinance, computes DD_3Y, MA50
# Writes data/prices_dd.parquet
```

**3. Sweep drawdown thresholds (training data only)**
```bash
python prices_sweep.py --train-end 2021-12-31
# Writes data/threshold_sector_summary_train.csv
```

**4. Select best threshold per sector**
```bash
python interpret_thresholds.py
# Writes data/best_thresholds_by_sector_train.csv
```

**5. Generate falling-knife events**
```bash
# Training events
python analyze.py --test-start 2015-01-01 --test-end 2021-12-31

# Test events
python analyze.py --test-start 2022-01-01 --test-end 2025-12-31
# Writes data/event_metrics_test.csv
```

**6. Optimize strategy on training data, evaluate on test**
```bash
python optimize_strategy.py
# Writes data/best_train_config.json, data/test_trades_best_config.csv
```

**7. Run frozen strategy on test window**
```bash
python backtest_confirm.py
# Writes data/run_trades.csv, data/run_metrics.csv
```

**8. Generate summaries and charts**
```bash
python results.py
python graph.py
```

---

## Signal Definition

A falling-knife event fires when a stock's closing price falls more than $\tau_k$ below its rolling 3-year high:

$$\mathrm{DD}_t = \frac{P_t}{\max_{s \in [t-755,\, t]} P_s} - 1 \leq -\tau_k$$

Sector-specific thresholds $\tau_k$ are learned on pre-2022 data using a composite score:

$$\text{Score} = 0.6\,\tilde{R} - 0.2\,\tilde{T} + 0.2\,\tilde{N}$$

where $\tilde{R}$ is normalized recovery rate, $\tilde{T}$ is normalized median recovery time, and $\tilde{N}$ is normalized event count.

**Selected thresholds (frozen pre-2022):**

| Sector | Threshold | Train Recovery Rate |
|---|---|---|
| Financial Services | 34% | 96.0% |
| Energy | 40% | 87.8% |
| Utilities | 27% | 86.3% |

---

## Strategy Rules

| Parameter | Value |
|---|---|
| Entry | Close of next trading day after trigger |
| Take-profit | +30% |
| Stop-loss | −12% |
| Max holding period | 120 trading days |
| Universe filter | Energy, Utilities, Financial Services; low trigger frequency tickers only |

---

## Data Notes

- **Source:** Daily OHLCV via `yfinance` (`auto_adjust=False`)
- **Universe:** Static S&P 500 snapshot — no survivorship bias correction
- **Prices:** Unadjusted closing prices (not corrected for dividends or splits)
- **Execution:** Next-period close assumption (signal at close of day $t$, entry at close of day $t+1$)
- The `data/` folder (parquet files, output CSVs) is excluded from this repo and must be regenerated locally by running the pipeline above

---

## Live Watchlist

To scan current S&P 500 constituents for active falling-knife signals:

```bash
python watchlist.py --tickers-file data/sp500_tickers.txt \
                    --best-thresholds data/best_thresholds_by_sector_train.csv
# Writes data/watchlist_signals.csv
```

---

## Citation

If you use this code or methodology, please cite:

```
Lazarakis, Angela. "Catching Falling Knives: A Systematic Study of Mean Reversion 
in Extreme Equity Drawdowns." St. George Capital, 2026.
```
