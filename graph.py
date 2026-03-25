import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# 1. Load Data
# -------------------------
events = pd.read_csv(
    "data/event_metrics_test.csv",
    parse_dates=["trigger_date", "bottom_date", "recovery_date", "ma50_cross_date"]
)
prices = pd.read_parquet("data/prices_dd.parquet")

# -------------------------
# 2. Summarize events by ticker
# -------------------------
summary = (
    events.groupby("ticker")
    .agg(
        n_events=("ticker", "count"),
        recovery_rate=("recovered_+20%", "mean"),
        median_days_to_bottom=("days_to_bottom", "median"),
        median_days_to_recovery=("days_bottom_to_recovery", "median"),
        median_days_to_ma50=("days_bottom_to_ma50", "median"),
        avg_drawdown=("drawdown_at_trigger", "mean")
    )
    .reset_index()
)

# Save summary
summary.to_csv("data/ticker_summary.csv", index=False)
print(summary.sort_values("recovery_rate", ascending=False).head())

# -------------------------
# 3. Plot all events for a single ticker
# -------------------------
tkr = "NEE"  # test w any ticker 
df = prices[prices["Ticker"] == tkr] 
evs = events[events["ticker"] == tkr]

# Ensure Date column is datetime
df["Date"] = pd.to_datetime(df["Date"])

plt.figure(figsize=(12,6))

# Plot price and moving average
plt.plot(df["Date"], df["Close"], label="Close")
plt.plot(df["Date"], df["MA50"], label="MA50", alpha=0.6)

# Plot all falling-knife events with proper legend
for i, ev in enumerate(evs.itertuples()):
    plt.axvline(ev.trigger_date, color="orange", ls="--", label="Trigger" if i==0 else "")
    plt.axvline(ev.bottom_date, color="red", ls="--", label="Bottom" if i==0 else "")
    if pd.notna(ev.recovery_date):
        plt.axvline(ev.recovery_date, color="green", ls="--", label="Recovery" if i==0 else "")

plt.title(f"All falling-knife events for {tkr}")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.show()