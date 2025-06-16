import matplotlib.pyplot as plt
from data_loader import fetch_stock_data, get_fundamentals, prime_cache_with_history
from merton_model import solve_kmv_model, solve_merton_model
from sensitivity import stress_test
from rolling_analysis import historical_quarterly_pd_series


def run_analysis(ticker: str, r: float = 0.04, T: float = 1.0, mode: str = "merton"):
    price, sigma_E = fetch_stock_data(ticker)
    shares_outstanding, total_debt, kmv_debt = get_fundamentals(ticker)
    E = price * shares_outstanding

    if mode == "kmv":
        result = solve_kmv_model(E, sigma_E, D_kmv=kmv_debt, r=r, T=T)
    else:
        result = solve_merton_model(E, sigma_E, D=total_debt, r=r, T=T)

    output = {
        "Ticker": ticker,
        "Price": price,
        "Shares Outstanding": shares_outstanding,
        "Equity Value": E,
        "Debt": kmv_debt if mode == "kmv" else total_debt,
        "Sigma_E": sigma_E,
        "V": result['V'],
        "Sigma_V": result['sigma_V'],
        "PD": result.get('PD_KMV') if mode == "kmv" else result['PD'],
        "DistanceToDefault": result.get('DistanceToDefault', None),
        "Mode": mode
    }

    for k, v in output.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    return output


def main_dispatch(ticker: str, analysis_type: str = "spot", mode: str = "kmv"):
    """
    Dispatch the selected analysis mode.
    - spot: one-shot static PD analysis
    - rolling: historical PD analysis (multi-year, rolling)
    """
    if analysis_type == "rolling":
        print(f"\n[INFO] Preparing 5-year historical data for {ticker}...")
        prime_cache_with_history(ticker, years=5)
        pd_series = historical_quarterly_pd_series(ticker, mode=mode)
        pd_series.plot(title=f"Rolling Default Probability - {ticker.upper()} ({mode.upper()})")
        plt.ylabel("Default Probability (PD)")
        plt.xlabel("Date")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print(f"\n[INFO] Running point-in-time analysis for {ticker}...")
        run_analysis(ticker, mode=mode)


if __name__ == "__main__":
    print("Merton / KMV Credit Risk Model")
    ticker = input("Enter ticker (e.g., AAPL): ").strip().upper()
    analysis_type = input("Select analysis type ('spot' or 'rolling'): ").strip().lower()
    mode = input("Select model ('merton' or 'kmv'): ").strip().lower()

    if analysis_type not in {"spot", "rolling"}:
        raise ValueError("Invalid analysis type. Please choose 'spot' or 'rolling'.")

    if mode not in {"merton", "kmv"}:
        raise ValueError("Invalid model. Please choose 'merton' or 'kmv'.")

    main_dispatch(ticker, analysis_type=analysis_type, mode=mode)
