import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_loader import load_or_download_data
from merton_model import solve_kmv_model, solve_merton_model

def historical_quarterly_pd_series(ticker: str, window: int = 63, mode: str = "kmv", T: float = 1.0, r: float = 0.04):
    tkr = yf.Ticker(ticker)
    balance_sheet = tkr.quarterly_balance_sheet.T
    balance_sheet.index = pd.to_datetime(balance_sheet.index)
    balance_sheet = balance_sheet.sort_index()

    relevant_cols = {
        "Total Debt": "total_debt",
        "Ordinary Shares Number": "shares_outstanding",
        "Long Term Debt": "long_term_debt",
        "Current Debt": "current_debt"
    }

    balance_sheet = balance_sheet[[col for col in relevant_cols if col in balance_sheet.columns]]
    balance_sheet = balance_sheet.rename(columns=relevant_cols)

    first_balance_date = balance_sheet.index.min()

    if pd.isna(first_balance_date):
        print(f"[WARN] No valid balance sheet date found for {ticker}. Using fallback start date.")
        first_balance_date = pd.Timestamp.today() - pd.DateOffset(years=5)

    price_start = (first_balance_date - pd.Timedelta(days=window)).strftime("%Y-%m-%d")
    price_end = (datetime.today().date() - timedelta(days=1)).strftime("%Y-%m-%d")
    prices_df = load_or_download_data(ticker, start=price_start, end=price_end)

    price_col = 'Adj Close' if 'Adj Close' in prices_df.columns else 'Close'
    prices = prices_df[price_col].dropna()

    print(f"[INFO] Prices available from {prices.index.min().date()} to {prices.index.max().date()}")

    pd_values = []
    pd_dates = []

    for date, row in balance_sheet.iterrows():
        price_window = prices.loc[prices.index <= date].tail(window)

        if len(price_window) < window * 0.8:
            print(f"[SKIP] Not enough data for window ending {date.date()} ({len(price_window)} points)")
            continue

        returns = np.log(price_window / price_window.shift(1)).dropna()
        sigma_E = returns.std() * np.sqrt(252)

        try:
            price = float(price_window.iloc[-1])
            shares_outstanding = float(row["shares_outstanding"])
            sigma_E = float(sigma_E)

            E = price * shares_outstanding

            if mode == "kmv" and "current_debt" in row and "long_term_debt" in row:
                D = float(row["current_debt"]) + 0.5 * float(row["long_term_debt"])
            else:
                D = float(row.get("total_debt"))

            if any(pd.isna([D, E, sigma_E])):
                print(f"[SKIP] Missing inputs at {date.date()} (D={D}, E={E}, sigma_E={sigma_E})")
                continue

            result = (
                solve_kmv_model(E, sigma_E, D_kmv=D, r=r, T=T)
                if mode == "kmv"
                else solve_merton_model(E, sigma_E, D=D, r=r, T=T)
            )
            pd_val = result.get("PD_KMV", result["PD"])

            pd_values.append(pd_val)
            pd_dates.append(date)

        except Exception as e:
            print(f"[ERROR] {ticker} @ {date.date()} → {e}")
            continue

    if not pd_values:
        print(f"[WARN] No valid PD points generated for {ticker}.")

    pd_df = pd.DataFrame({"Date": pd_dates, "PD": pd_values})
    return pd_df.set_index("Date")

