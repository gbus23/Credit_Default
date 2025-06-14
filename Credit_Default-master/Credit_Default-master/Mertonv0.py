import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import root


def download_stock_data(ticker: str, period: str = "1y"):
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)

    if hist.empty:
        raise ValueError(f"Pas de données disponibles pour le ticker {ticker}.")

    price_col = 'Adj Close' if 'Adj Close' in hist.columns else 'Close'
    prices = hist[price_col].dropna()
    price = float(prices.iloc[-1].item())
    returns = np.log(prices / prices.shift(1)).dropna()
    sigma_E = float(returns.std()) * np.sqrt(252)

    return price, sigma_E


def get_fundamentals(ticker: str):
    stock = yf.Ticker(ticker)
    info = stock.info

    shares_outstanding = info.get("sharesOutstanding")
    total_debt = info.get("totalDebt")

    if shares_outstanding is None or total_debt is None:
        raise ValueError(f"Impossible de récupérer les fondamentaux pour {ticker}.")

    return shares_outstanding, total_debt


def merton_equations(x, E, sigma_E, D, r, T):
    V, sigma_V = x
    d1 = (np.log(V / D) + (r + 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))
    d2 = d1 - sigma_V * np.sqrt(T)

    eq1 = V * norm.cdf(d1) - D * np.exp(-r * T) * norm.cdf(d2) - E
    eq2 = sigma_V * V * norm.cdf(d1) - sigma_E * E
    return [eq1, eq2]


def solve_merton_model(E, sigma_E, D, r=0.04, T=1.0):
    initial_guess = [E + D, 0.3]
    sol = root(merton_equations, initial_guess, args=(E, sigma_E, D, r, T))

    if not sol.success:
        raise RuntimeError(f"Échec de la résolution du modèle de Merton : {sol.message}")

    V, sigma_V = sol.x
    d2 = (np.log(V / D) + (r - 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))
    pd = norm.cdf(-d2)

    return {
        "V": V,
        "sigma_V": sigma_V,
        "PD": pd
    }


def analyze_firm(ticker: str, r: float = 0.04, T: float = 1.0):
    price, sigma_E = download_stock_data(ticker)
    shares_outstanding, debt = get_fundamentals(ticker)
    E = price * shares_outstanding

    result = solve_merton_model(E, sigma_E, D=debt, r=r, T=T)

    print(f"--- Analyse Merton pour {ticker} ---")
    print(f"Prix de l'action : {price:.2f} USD")
    print(f"Actions en circulation : {shares_outstanding / 1e6:.2f} M")
    print(f"Capitalisation boursière (E) : {E/1e9:.2f} Md USD")
    print(f"Dette (D) : {debt/1e9:.2f} Md USD")
    print(f"Volatilité action (sigma_E) : {sigma_E:.2%}")
    print(f"Valeur estimée des actifs (V) : {result['V']/1e9:.2f} Md USD")
    print(f"Volatilité des actifs (sigma_V) : {result['sigma_V']:.2%}")
    print(f"Probabilité de défaut à 1 an : {result['PD']:.2%}")

    return result


analyze_firm("AAL")
