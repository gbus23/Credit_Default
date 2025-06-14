
import numpy as np
from scipy.stats import norm
from scipy.optimize import root

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

def solve_kmv_model(E, sigma_E, D_kmv, r=0.04, T=1.0):
    result = solve_merton_model(E, sigma_E, D=D_kmv, r=r, T=T)
    V, sigma_V = result['V'], result['sigma_V']

    dd = (np.log(V / D_kmv) + (r - 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))
    pd_kmv = norm.cdf(-dd)

    result['DistanceToDefault'] = dd
    result['PD_KMV'] = pd_kmv
    return result
