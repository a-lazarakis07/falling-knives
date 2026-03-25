import pandas as pd

df = pd.read_csv("sp500.csv")   # <-- replace filename
tickers = df.iloc[:, 0].dropna().astype(str).str.upper()

tickers.to_csv("data/sp500_tickers.txt", index=False, header=False)
print("Saved to data/sp500_tickers.txt")