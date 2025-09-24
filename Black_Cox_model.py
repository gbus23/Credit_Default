import numpy as np
from scipy.stats import norm
from merton_model import solve_merton_model

def black_cox_pd(E, sigma_E, D, B=None, r=0.04, T=1.0, mu=None, verbose=False):
    """
    Compute Black–Cox default probability.
    
    Parameters
    ----------
    E : float
        Market value of equity.
    sigma_E : float
        Equity volatility.
    D : float
        Total debt (or proxy).
    B : float, optional
        Default barrier. Default = 0.7 * D if not provided.
    r : float, optional
        Risk-free rate. Default 0.04.
    T : float, optional
        Time horizon in years. Default 1.0.
    mu : float, optional
        Asset drift. Default = r.
    verbose : bool, optional
        Print solver warnings if True.

    Returns
    -------
    dict
        {
            "V": asset value,
            "sigma_V": asset volatility,
            "PD_BC": Black–Cox default probability,
            "Survival": survival probability
        }
    """
    # Step 1: solve asset value and volatility using Merton
    sol = solve_merton_model(E, sigma_E, D, r, T, verbose)
    V, sigma_V = sol["V"], sol["sigma_V"]

    if np.isnan(V) or np.isnan(sigma_V):
        return {"PD_BC": np.nan}

    # Step 2: drift
    if mu is None:
        mu = r

    # Step 3: barrier
    if B is None:
        B = 0.7 * D

    # Step 4: Black–Cox survival probability
    numerator = np.log(V / B)
    denom = sigma_V * np.sqrt(T)
    drift = (mu - 0.5 * sigma_V**2) * T

    d1 = (numerator + drift) / denom
    d2 = (numerator - drift) / denom
    lam = (mu - 0.5 * sigma_V**2) / sigma_V**2
    term = (B / V) ** (2 * lam)

    survival = norm.cdf(d1) - term * norm.cdf(d2)
    pd_bc = 1 - survival

    return {
        "V": V,
        "sigma_V": sigma_V,
        "PD_BC": pd_bc,
        "Survival": survival
    }
