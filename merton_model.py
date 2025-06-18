import numpy as np
from scipy.stats import norm
from scipy.optimize import root, minimize_scalar

def merton_equations(x, E, sigma_E, D, r, T):
    V, sigma_V = x
    if V <= 0 or sigma_V <= 0:
        return [1e6, 1e6]  # force divergence
    d1 = (np.log(V / D) + (r + 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))
    d2 = d1 - sigma_V * np.sqrt(T)
    eq1 = V * norm.cdf(d1) - D * np.exp(-r * T) * norm.cdf(d2) - E
    eq2 = sigma_V * V * norm.cdf(d1) - sigma_E * E
    return [eq1, eq2]

def solve_merton_model(E, sigma_E, D, r=0.04, T=1.0, verbose=False):
    if E <= 0 or sigma_E <= 0 or D <= 0:
        return {"V": np.nan, "sigma_V": np.nan, "PD": np.nan, "DistanceToDefault": np.nan}

    initial_guess = [E + D, 0.3]

    sol = root(merton_equations, initial_guess, args=(E, sigma_E, D, r, T), method='hybr')

    if not sol.success:
        if verbose:
            print(f"[WARNING] Root solver failed: {sol.message}")
        return {"V": np.nan, "sigma_V": np.nan, "PD": np.nan, "DistanceToDefault": np.nan}

    V, sigma_V = sol.x
    if V <= 0 or sigma_V <= 0:
        return {"V": np.nan, "sigma_V": np.nan, "PD": np.nan, "DistanceToDefault": np.nan}

    d2 = (np.log(V / D) + (r - 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))
    pd = norm.cdf(-d2)

    return {
        "V": V,
        "sigma_V": sigma_V,
        "PD": pd,
        "DistanceToDefault": d2
    }

def solve_kmv_model(E, sigma_E, D_kmv, r=0.04, T=1.0, verbose=False):
    result = solve_merton_model(E, sigma_E, D=D_kmv, r=r, T=T, verbose=verbose)
    result["PD_KMV"] = result.get("PD", np.nan)
    return result
