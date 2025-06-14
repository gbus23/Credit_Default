import numpy as np
import pandas as pd
import yfinance as yf
import os
import pickle
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

CACHE_DIR = "cache_data"
os.makedirs(CACHE_DIR, exist_ok=True)

def cache_path(ticker: str) -> str:
    """Retourne le chemin du fichier cache pour un ticker."""
    return os.path.join(CACHE_DIR, f"{ticker}_cache.pkl")

def load_or_download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Charge les données depuis le cache si elles couvrent la période souhaitée,
    sinon télécharge les données manquantes et met à jour le cache.
    """
    end_dt = pd.to_datetime(end).date()
    adjusted_end = (end_dt - timedelta(days=1)).strftime("%Y-%m-%d")
    path = cache_path(ticker)

    request_start = pd.to_datetime(start).date()
    request_end = pd.to_datetime(adjusted_end).date()

    print(f"Requesting data from {request_start} to {request_end}")
    print(f"Cache path: {path}")

    if os.path.exists(path):
        with open(path, "rb") as f:
            cached = pickle.load(f)

        cached.index = pd.to_datetime(cached.index)
        cached_start = cached.index.min().date()
        cached_end = cached.index.max().date()

        print(f"Cache range: {cached_start} to {cached_end}")
        print(f"[DEBUG] Index dates in cache: min={cached.index.min()}, max={cached.index.max()}")

        covers_range = (
            request_start >= cached_start - timedelta(days=2) and
            request_end <= cached_end + timedelta(days=2)
        )

        if covers_range:
            print(f"[CACHE] Using cached data for {ticker} ({request_start} to {request_end})")
            return cached.loc[
                (cached.index.date >= request_start) & (cached.index.date <= request_end)
            ]

        print(f"[CACHE] Incomplete cache for {ticker}.")
        full_data = cached

        if request_start < cached_start:
            print(f"[CACHE] Downloading missing data BEFORE cache: ({request_start} to {cached_start - timedelta(days=1)})")
            before = yf.download(ticker, start=request_start, end=cached_start)
            full_data = pd.concat([before, full_data])

        if request_end > cached_end:
            print(f"[CACHE] Downloading missing data AFTER cache: ({cached_end + timedelta(days=1)} to {request_end})")
            after = yf.download(ticker, start=cached_end + timedelta(days=1), end=request_end + timedelta(days=1))
            full_data = pd.concat([full_data, after])

        full_data = full_data.drop_duplicates()
        with open(path, "wb") as f:
            pickle.dump(full_data, f)

        return full_data.loc[
            (full_data.index.date >= request_start) & (full_data.index.date <= request_end)
        ]

    print(f"[DOWNLOAD] No cache found. Downloading full data for {ticker} ({request_start} to {request_end})")
    df = yf.download(ticker, start=request_start, end=request_end + timedelta(days=1))
    if df.empty:
        raise ValueError(f"No data available for {ticker} between {start} and {adjusted_end}.")
    with open(path, "wb") as f:
        pickle.dump(df, f)
    return df


def get_fundamentals(ticker: str):
    """
    Récupère les données fondamentales globales les plus récentes via yfinance.
    """
    print(f"[DOWNLOAD] Downloading fundamentals for {ticker}")
    stock = yf.Ticker(ticker)
    info = stock.info

    shares_outstanding = info.get("sharesOutstanding")
    total_debt = info.get("totalDebt")
    current_liabilities = info.get("currentDebt")
    long_term_debt = info.get("longTermDebt")

    if shares_outstanding is None or total_debt is None:
        raise ValueError(f"Missing fundamental data for {ticker}.")

    if current_liabilities is not None and long_term_debt is not None:
        kmv_debt = current_liabilities + 0.5 * long_term_debt
    else:
        kmv_debt = total_debt

    return shares_outstanding, total_debt, kmv_debt


def fetch_stock_data(ticker: str, years: int = 1):
    """
    Calcule le prix actuel et la volatilité historique de l’action.
    """
    end = datetime.today().date()
    start = (end - relativedelta(years=years)).strftime("%Y-%m-%d")
    end = end.strftime("%Y-%m-%d")
    data = load_or_download_data(ticker, start, end)

    price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
    prices = data[price_col].dropna()
    price = float(prices.iloc[-1].item())
    returns = np.log(prices / prices.shift(1)).dropna()
    sigma_E = float(returns.std()) * np.sqrt(252)

    return price, sigma_E


def prime_cache_with_history(ticker: str, years: int = 5):
    """
    Précharge les données de prix sur plusieurs années pour initialiser le cache.
    """
    end = datetime.today().date()
    start = (end - relativedelta(years=years)).strftime("%Y-%m-%d")
    end = end.strftime("%Y-%m-%d")
    _ = load_or_download_data(ticker, start, end)
