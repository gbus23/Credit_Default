# sensitivity.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from merton_model import solve_merton_model, solve_kmv_model
from data_loader import load_or_download_data, get_fundamentals,fetch_stock_data
from datetime import datetime, timedelta

def stress_test(ticker: str, mode: str = "kmv", r: float = 0.04, T: float = 1.0):
    # Use yesterday to ensure available data
    end_date = datetime.today().date() - timedelta(days=1)
    start_date = end_date - timedelta(days=252)  # 1 year of data

    prices_df = load_or_download_data(ticker, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
    price_col = 'Adj Close' if 'Adj Close' in prices_df.columns else 'Close'
    prices = prices_df[price_col].dropna()

    # Latest price and annualized volatility
    price = float(prices.iloc[-1])
    returns = np.log(prices / prices.shift(1)).dropna()
    sigma_E = float(returns.std()) * np.sqrt(252)

    # Fundamentals
    shares_outstanding, total_debt, kmv_debt = get_fundamentals(ticker)
    E = price * shares_outstanding
    D = kmv_debt if mode == "kmv" else total_debt

    sigma_range = np.linspace(0.1, 0.8, 10)
    debt_range = np.linspace(D * 0.5, D * 1.5, 10)

    results = []

    for sigma in sigma_range:
        for debt in debt_range:
            try:
                if mode == "kmv":
                    res = solve_kmv_model(E, sigma, D_kmv=debt, r=r, T=T)
                    pd_value = res['PD_KMV']
                else:
                    res = solve_merton_model(E, sigma, D=debt, r=r, T=T)
                    pd_value = res['PD']
                results.append({"Sigma_E": sigma, "Debt": debt, "PD": pd_value})
            except Exception:
                results.append({"Sigma_E": sigma, "Debt": debt, "PD": np.nan})

    df = pd.DataFrame(results)
    pivot = df.pivot(index="Sigma_E", columns="Debt", values="PD")

    plt.figure(figsize=(10, 6))
    contour = plt.contourf(pivot.columns, pivot.index, pivot.values, levels=20, cmap="viridis")
    plt.colorbar(contour, label="Probability of Default")
    plt.xlabel("Debt")
    plt.ylabel("Equity Volatility (Sigma_E)")
    plt.title(f"Stress Test - {ticker.upper()} ({mode.upper()})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return df
