# Merton Model (1974) – Summary Sheet

## 1. Overview

The **Merton model** is a structural credit risk model in which:

* A firm's **equity** is modeled as a **European call option** on the value of its total assets.
* **Default** occurs at debt maturity $T$ if the firm's asset value $V(T)$ is less than the debt face value $D$.
* **Creditors** are repaid in full if $V(T) \geq D$; otherwise, they receive the remaining asset value $V(T)$.

## 2. Key Assumptions

* The firm's total asset value follows a **geometric Brownian motion**:

  $$
  dV(t) = \mu V(t)\,dt + \sigma V(t)\,dW(t)
  $$

  where:

  * $\mu$: expected return of the asset
  * $\sigma$: asset volatility
  * $W(t)$: standard Brownian motion

* A single zero-coupon debt with face value $D$, maturing at time $T$.

* No taxes, no transaction costs, no arbitrage, and a constant risk-free rate $r$.

## 3. Mathematical Formulation

### 3.1 Equity Value (as a call option)

$$
S_0 = V_0 N(d_1) - D e^{-rT} N(d_2)
$$

where:

$$
d_1 = \frac{\ln(V_0/D) + (r + \frac{1}{2}\sigma^2)T}{\sigma \sqrt{T}}, \quad
d_2 = d_1 - \sigma \sqrt{T}
$$

* $V_0$: current value of the firm's assets
* $N(\cdot)$: cumulative distribution function of the standard normal distribution

### 3.2 Default Probability at Maturity

$$
\mathbb{P}(\text{default at } T) = \mathbb{P}(V(T) < D) = N(-d_2)
$$

### 3.3 Debt Value

$$
\text{Debt}_0 = V_0 - S_0 = D e^{-rT} N(d_2) + \text{Expected loss in case of default}
$$

## 4. Economic Interpretation

| Concept       | Interpretation                                                       |
| ------------- | -------------------------------------------------------------------- |
| Equity        | Call option on the firm's asset value                                |
| Default event | Occurs if $V(T) < D$ at maturity                                     |
| Default prob. | Equal to $N(-d_2)$, the left tail of the log-normal distribution     |
| Creditors     | Effectively long a risk-free bond and short a put option on the firm |

## 5. Applications

* Estimating **default probabilities** based on market information (e.g., equity prices, volatility).
* Computing the **value of risky debt** and the associated **credit spread**.
* Basis for more advanced structural credit models (e.g., Black–Cox, Leland–Toft).


## 6. Project Structure  
## Project Structure  

## Project Structure  

Credit_Default/  
│  
├── cache_data/           # Cached data files to avoid redundant computations  
├── test/                 # Unit tests for model validation and consistency checks  
│  
├── data_loader.py        # Data loading and preprocessing utilities  
├── main.py               # Main script for console-based execution  
├── main_dash.py          # Interactive dashboard (Dash) for visualization  
├── merton_model.py       # Core implementation of the Merton structural credit risk model  
├── rolling_analysis.py   # Rolling window analysis for time series evaluation  
├── sensitivity.py        # Sensitivity analysis on model parameters  
├── stress_test.py        # Stress testing under adverse market scenarios  
└── README.md             # Project documentation  

