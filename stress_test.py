import numpy as np
from merton_model import solve_merton_model

def stress_test(E, sigma_E, D, r=0.04, T=1.0, shocks=None):
    """
    Applique des chocs de marché au modèle de Merton et renvoie les métriques recalculées.

    Parameters:
        E (float): Market value of equity
        sigma_E (float): Volatility of equity
        D (float): Debt
        r (float): Risk-free rate
        T (float): Time horizon in years
        shocks (dict): Dictionnaire des chocs à appliquer, par ex :
                       {"E": 0.8, "sigma_E": 1.5, "r": 0.05}

    Returns:
        dict: Résultats du modèle avec les paramètres choqués
    """
    if shocks is None:
        shocks = {}

    E_shock = E * shocks.get("E", 1.0)
    sigma_E_shock = sigma_E * shocks.get("sigma_E", 1.0)
    D_shock = D * shocks.get("D", 1.0)
    r_shock = shocks.get("r", r)
    T_shock = shocks.get("T", T)

    try:
        result = solve_merton_model(E_shock, sigma_E_shock, D_shock, r=r_shock, T=T_shock)
        result.update({
            "E_shock": E_shock,
            "sigma_E_shock": sigma_E_shock,
            "D_shock": D_shock,
            "r_shock": r_shock,
            "T_shock": T_shock
        })
        return result
    except Exception as e:
        return {"error": str(e)}
