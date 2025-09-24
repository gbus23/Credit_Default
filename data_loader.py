import numpy as np
import pandas as pd
import yfinance as yf
import os
import pickle
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

CACHE_DIR = "cache_data"
os.makedirs(CACHE_DIR, exist_ok=True)

# --------------------------------------------------------------------
# Cache utilities
# --------------------------------------------------------------------
def cache_path(ticker: str) -> str:
    """Return cache file path for a given ticker."""
    return os.path.join(CACHE_DIR, f"{ticker}_cache.pkl")


def load_or_download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Load price data from cache if available, otherwise download from Yahoo Finance
    and update the cache.
    """
    end_dt = pd.to_datetime(end).date()
    adjusted_end = (end_dt - timedelta(days=1)).strftime("%Y-%m-%d")
    path = cache_path(ticker)

    request_start = pd.to_datetime(start).date()
    request_end = pd.to_datetime(adjusted_end).date()

    if os.path.exists(path):
        with open(path, "rb") as f:
            cached = pickle.load(f)

        cached.index = pd.to_datetime(cached.index)
        cached_start = cached.index.min().date()
        cached_end = cached.index.max().date()

        covers_range = (
            request_start >= cached_start - timedelta(days=2)
            and request_end <= cached_end + timedelta(days=2)
        )

        if covers_range:
            return cached.loc[
                (cached.index.date >= request_start)
                & (cached.index.date <= request_end)
            ]

        full_data = cached

        if request_start < cached_start:
            before = yf.download(ticker, start=request_start, end=cached_start)
            full_data = pd.concat([before, full_data])

        if request_end > cached_end:
            after = yf.download(
                ticker,
                start=cached_end + timedelta(days=1),
                end=request_end + timedelta(days=1),
            )
            full_data = pd.concat([full_data, after])

        full_data = full_data.drop_duplicates()
        with open(path, "wb") as f:
            pickle.dump(full_data, f)

        return full_data.loc[
            (full_data.index.date >= request_start)
            & (full_data.index.date <= request_end)
        ]

    df = yf.download(ticker, start=request_start, end=request_end + timedelta(days=1))
    if df.empty:
        raise ValueError(f"No data available for {ticker} between {start} and {adjusted_end}.")
    with open(path, "wb") as f:
        pickle.dump(df, f)
    return df

# --------------------------------------------------------------------
# Debt snapshot helper
# --------------------------------------------------------------------
ST_ALIASES = [
    "Short Long Term Debt", "Current Debt", "Current Portion Of Long Term Debt",
    "Current Portion of Long Term Debt", "Short Term Debt", "Short Term Borrowings"
]
LT_ALIASES = ["Long Term Debt", "Long-Term Debt", "Long Term Borrowings"]
TOT_ALIASES = ["Total Debt", "TotalDebt"]


def _first_non_nan(df, labels, col):
    for lab in labels:
        if lab in df.index:
            val = df.loc[lab, col]
            if pd.notna(val):
                return float(val)
    return None


def get_debt_snapshot(ticker: str, prefer_quarterly: bool = True):
    """
    Returns (st_debt, lt_debt, total_debt, asof_date, source).
    Prefers quarterly balance sheet; falls back to annual; fills missing pieces.
    """
    t = yf.Ticker(ticker)
    bs = None
    source = None

    if prefer_quarterly:
        try:
            q = t.quarterly_balance_sheet
            if q is not None and not q.empty:
                bs, source = q, "quarterly_balance_sheet"
        except Exception:
            pass

    if bs is None:
        try:
            a = t.balance_sheet
            if a is not None and not a.empty:
                bs, source = a, "balance_sheet"  # annual
        except Exception:
            pass

    st = lt = tot = None
    asof = None

    if bs is not None and not bs.empty:
        # Always pick the most recent available column
        latest_col = max(bs.columns)
        asof = pd.to_datetime(latest_col).date()
        st = _first_non_nan(bs, ST_ALIASES, latest_col)
        lt = _first_non_nan(bs, LT_ALIASES, latest_col)
        tot = _first_non_nan(bs, TOT_ALIASES, latest_col)

    # Fallback to info.totalDebt if missing
    if tot is None:
        tot = (t.info or {}).get("totalDebt")

    # Reconstruct missing legs
    if st is None and lt is not None and tot is not None:
        st = max(tot - lt, 0.0)
    if lt is None and st is not None and tot is not None:
        lt = max(tot - st, 0.0)

    print(f"[DEBUG] Debt snapshot for {ticker}: asof={asof}, source={source}, ST={st}, LT={lt}, TOT={tot}")
    return st, lt, tot, asof, source

# --------------------------------------------------------------------
# Fundamentals
# --------------------------------------------------------------------
def get_fundamentals(ticker: str):
    """
    Retrieve fundamentals including shares outstanding and detailed debt breakdown.
    Returns:
        shares_outstanding, total_debt, short_term_debt, long_term_debt, debt_asof, debt_source
    """
    print(f"[DOWNLOAD] Downloading fundamentals for {ticker}")
    stock = yf.Ticker(ticker)

    # Shares outstanding always from info
    info = stock.info
    shares_outstanding = info.get("sharesOutstanding")
    if shares_outstanding is None:
        raise ValueError(f"Missing sharesOutstanding for {ticker}.")

    short_term_debt, long_term_debt, total_debt, debt_asof, debt_source = get_debt_snapshot(ticker)

    if total_debt is None:
        raise ValueError(f"Missing debt data for {ticker}.")

    return shares_outstanding, total_debt, short_term_debt, long_term_debt, debt_asof, debt_source

# --------------------------------------------------------------------
# Equity price & volatility
# --------------------------------------------------------------------
def fetch_stock_data(ticker: str, years: int = 1):
    """
    Compute current equity price and historical volatility.
    Falls back to yfinance.info if price data is missing.
    """
    end = datetime.today().date()
    start = (end - relativedelta(years=years)).strftime("%Y-%m-%d")
    end = end.strftime("%Y-%m-%d")

    data = yf.download(ticker, start=start, end=end)

    if data.empty:
        info = yf.Ticker(ticker).info
        price = info.get("regularMarketPrice")
        if price is None:
            raise ValueError(f"No data or price info available for {ticker}")
        sigma_E = info.get("beta", 1.0) * 0.2  # very rough proxy
        return price, sigma_E

    price_col = "Adj Close" if "Adj Close" in data.columns else "Close"
    prices = data[price_col].dropna()

    if prices.empty:
        raise ValueError(f"No price series available for {ticker} in {price_col}")

    price = float(prices.iloc[-1])
    returns = np.log(prices / prices.shift(1)).dropna()
    sigma_E = float(returns.std()) * np.sqrt(252)

    return price, sigma_E

# --------------------------------------------------------------------
# Cache priming
# --------------------------------------------------------------------
def prime_cache_with_history(ticker: str, years: int = 5):
    """
    Pre-load multiple years of price history to warm up cache.
    """
    end = datetime.today().date()
    start = (end - relativedelta(years=years)).strftime("%Y-%m-%d")
    end = end.strftime("%Y-%m-%d")
    _ = load_or_download_data(ticker, start, end)
