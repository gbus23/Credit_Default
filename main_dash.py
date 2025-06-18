import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import yfinance as yf

from data_loader import fetch_stock_data, get_fundamentals, prime_cache_with_history
from merton_model import solve_kmv_model, solve_merton_model
from rolling_analysis import historical_quarterly_pd_series
from stress_test import stress_test

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

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

    return pd.DataFrame.from_dict(output, orient='index', columns=['Value'])

def get_company_summary(ticker_str: str) -> str:
    try:
        ticker = yf.Ticker(ticker_str)
        info = ticker.info

        name = info.get("longName", ticker_str)
        sector = info.get("sector", "N/A")
        industry = info.get("industry", "N/A")
        employees = info.get("fullTimeEmployees", None)
        market_cap = info.get("marketCap", None)

        summary = f"{name} operates in the {sector.lower()} sector, specifically in the {industry.lower()} industry."

        if employees:
            summary += f" It employs approximately {employees:,} people"
        if market_cap:
            mc = f"${market_cap / 1e9:.1f}B" if market_cap > 1e9 else f"${market_cap / 1e6:.1f}M"
            summary += f" and has a market capitalization of about {mc}."

        return summary 
    except Exception:
        return f"No description available for {ticker_str.upper()}."

def get_rolling_plot(ticker: str, mode: str):
    prime_cache_with_history(ticker, years=5)

    pd_series = historical_quarterly_pd_series(ticker, mode=mode)

    if isinstance(pd_series, pd.DataFrame) and 'PD' in pd_series.columns:
        pd_series = pd_series["PD"]

    pd_series = pd_series.replace([np.inf, -np.inf], np.nan).dropna()
    pd_series = pd_series[~pd_series.index.duplicated(keep='first')]
    pd_series.index = pd.to_datetime(pd_series.index, errors='coerce')
    pd_series = pd_series[~pd_series.index.isna()]
    pd_series = pd_series.sort_index()

    if pd_series.empty:
        return go.Figure(layout={
            "title": f"Aucune donnée disponible pour {ticker.upper()} ({mode.upper()})",
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
        })

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd_series.index,
        y=pd_series.values,
        mode='lines+markers',
        name='PD',
        line=dict(width=2),
        hovertemplate='%{x|%b %Y}<br>PD: %{y:.2%}<extra></extra>',
    ))

    fig.update_layout(
        title=f"{ticker.upper()} – PD ({mode.upper()})",
        xaxis_title="Date",
        yaxis_title="Probability of Default",
        xaxis=dict(
            tickformat="%Y-%m",
            tickangle=45,
            tickmode='auto',
            nticks=10,
            showgrid=True
        ),
        yaxis=dict(showgrid=True),
        template="plotly_white",
        height=400,
        width=800
    )

    return fig


def generate_stress_heatmap(E, sigma_E, D, r, T=1.0):
    sigma_shocks = np.arange(-50, 105, 5)  # -50% à +100%
    debt_shocks = np.arange(-50, 105, 5)
    pd_matrix = np.zeros((len(sigma_shocks), len(debt_shocks)))

    for i, s_pct in enumerate(sigma_shocks):
        for j, d_pct in enumerate(debt_shocks):
            s_factor = 1 + s_pct / 100
            d_factor = 1 + d_pct / 100
            try:
                result = solve_merton_model(
                    E=E,
                    sigma_E=sigma_E * s_factor,
                    D=D * d_factor,
                    r=r,
                    T=T
                )
                pd_matrix[i, j] = result["PD"]
            except Exception:
                pd_matrix[i, j] = np.nan

    x_labels = [f"{d:+}%" for d in debt_shocks]
    y_labels = [f"{s:+}%" for s in sigma_shocks]

    fig = go.Figure(data=go.Heatmap(
        z=pd_matrix,
        x=x_labels,
        y=y_labels,
        colorscale="Viridis",
        colorbar=dict(title="PD", ticksuffix="%", tickformat=".1f"),
        hovertemplate="ΔDebt: %{x}<br>ΔVolatility: %{y}<br>PD: %{z:.2%}<extra></extra>"
    ))

    fig.update_layout(
        title="Stress Test Heatmap: PD vs Volatility & Debt Change",
        xaxis_title="Change in Debt",
        yaxis_title="Change in Volatility",
        height=600,
        width=800,
        template="plotly_white",
        font=dict(size=14),
        margin=dict(l=60, r=30, t=60, b=60)
    )

    return fig


app.layout = dbc.Container([
    html.H1("Merton / KMV Credit Risk Model", className="text-center my-4"),

    # Input section
    dbc.Row([
        dbc.Col([
            dbc.Input(id="ticker", placeholder="Enter ticker (e.g. AAPL)", type="text", className="mb-2"),
            dcc.Dropdown(id="mode", options=[
                {"label": "Merton Model", "value": "merton"},
                {"label": "KMV Model", "value": "kmv"}
            ], placeholder="Select model", className="mb-3"),
            dbc.Button("Run Analysis", id="run_button", color="primary", className="w-100")
        ], width=6)
    ], justify="center", className="mb-4"),

    # Company description
    dbc.Row([
            dbc.Col(html.Div(id="company_description"), width=8)
        ], justify="center", className="mb-4"),

        dbc.Row([
        dbc.Col([
            html.H5("Financial Summary", className="text-center mb-3"),
            dbc.Card([
                dbc.CardBody(html.Div(id="output_table"))
            ], className="mb-4")
        ])
    ], justify="center"),

    # Graphe PD en dessous
    dbc.Row([
        dbc.Col(dcc.Loading(dcc.Graph(id="rolling_plot")))
    ], justify="center"),

    # Stress test parameters
    html.H4("Stress Test Parameters", className="text-center mt-4"),
    dbc.Row([
        dbc.Col([
            html.Label("Change in Equity Value (%)"),
            dbc.Input(id="shock_E", type="number", value=0, className="mb-3"),
            html.Label("Change in Equity Volatility (%)"),
            dbc.Input(id="shock_sigma", type="number", value=0, className="mb-3"),
            html.Label("Change in Debt (%)"),
            dbc.Input(id="shock_D", type="number", value=0, className="mb-3"),
            html.Label("Change in Risk-Free Rate (bps)"),
            dbc.Input(id="shock_r", type="number", value=0, className="mb-3"),
            dbc.Button("Apply Stress Test", id="stress_button", color="danger", className="w-100")
        ], width=6)
    ], justify="center", className="mb-4"),

    # Stress test output table
    dbc.Row([
        dbc.Col(html.Div(id="stress_table"), width=6)
    ], justify="center", className="mb-4"),

    # Heatmap output
    dbc.Row([
        dbc.Col(dcc.Loading(dcc.Graph(id="stress_heatmap")), width=8)
    ], justify="center", className="mb-4")

], fluid=True)


@app.callback(
    Output("output_table", "children"),
    Output("rolling_plot", "figure"),
    Output("company_description", "children"),
    Input("run_button", "n_clicks"),
    State("ticker", "value"),
    State("mode", "value")
)
def update_rolling_output(n_clicks, ticker, mode):
    if not n_clicks or not ticker or not mode:
        return None, go.Figure(), None

    df = run_analysis(ticker, mode=mode)
    raw = df.to_dict()["Value"]

    clean_labels = {
        "Ticker": "Ticker",
        "Price": "Price",
        "Shares Outstanding": "Shares Outstanding",
        "Equity Value": "Equity Value",
        "Debt": "Debt",
        "Sigma_E": "Equity Volatility (σ_E)",
        "V": "Firm Value (V)",
        "Sigma_V": "Asset Volatility (σ_V)",
        "PD": "Probability of Default",
        "DistanceToDefault": "Distance to Default",
        "Mode": "Model"
    }

    formatted = {}
    for key, val in raw.items():
        label = clean_labels.get(key, key)
        if isinstance(val, float):
            if "Probability" in label or "Volatility" in label:
                formatted[label] = f"{val:.2%}"
            elif label in {"Equity Value", "Firm Value (V)", "Equity after Shock", "Price"}:
                formatted[label] = f"${val:,.2f}"
            else:
                formatted[label] = round(val, 4)
        elif isinstance(val, int):
            if "Shares" in label:
                formatted[label] = f"{val:,}"
            elif "Debt" in label:
                formatted[label] = f"${val:,.2f}"
            else:
                formatted[label] = val
        else:
            formatted[label] = val

    df_formatted = pd.DataFrame(list(formatted.items()), columns=["Metric", "Value"])
    table = dbc.Table.from_dataframe(df_formatted, striped=True, bordered=True, hover=True)

    description = get_company_summary(ticker)
    description_card = dbc.Card([
        dbc.CardHeader("Company Overview"),
        dbc.CardBody(html.P(description, className="card-text"))
    ])

    fig = get_rolling_plot(ticker, mode)

    return table, fig, description_card


@app.callback(
    Output("stress_table", "children"),
    Output("stress_heatmap", "figure"),
    Input("stress_button", "n_clicks"),
    State("ticker", "value"),
    State("mode", "value"),
    State("shock_E", "value"),
    State("shock_sigma", "value"),
    State("shock_D", "value"),
    State("shock_r", "value")
)
def update_stress_test(n_clicks, ticker, mode, shock_E, shock_sigma, shock_D, shock_r):
    if not n_clicks or not ticker or not mode:
        return None, go.Figure()

    df = run_analysis(ticker, mode=mode)
    base = df["Value"].to_dict()

    result = stress_test(
        E=base["Equity Value"],
        sigma_E=base["Sigma_E"],
        D=base["Debt"],
        r=0.04,
        T=1.0,
        shocks={
            "E": 1 + (shock_E or 0) / 100,
            "sigma_E": 1 + (shock_sigma or 0) / 100,
            "D": 1 + (shock_D or 0) / 100,
            "r": 0.04 * (1 + (shock_r or 0) / 100)
        }
    )

    mapping = {
        "Firm Value (V)": ("V", "V"),
        "Asset Volatility (σ_V)": ("Sigma_V", "sigma_V"),
        "Probability of Default": ("PD", "PD"),
        "Distance to Default": ("DistanceToDefault", "DistanceToDefault"),
        "Equity Value": ("Equity Value", "E_shock"),
        "Equity Volatility (σ_E)": ("Sigma_E", "sigma_E_shock"),
        "Debt": ("Debt", "D_shock"),
        "Risk-Free Rate": ("r", "r_shock"),
        "Time Horizon": ("T", "T_shock")
    }

    def fmt(val, is_pct=False):
        if val is None:
            return "-"
        if isinstance(val, float):
            if is_pct:
                return f"{val:.2%}"
            elif val > 1e3:
                return f"${val:,.0f}"
            else:
                return round(val, 4)
        elif isinstance(val, int):
            return f"${val:,}"
        return str(val)

    rows = []
    for label, (before_key, after_key) in mapping.items():
        is_pct = "Volatility" in label or "Probability" in label or "Rate" in label
        before = fmt(base.get(before_key), is_pct)
        after = fmt(result.get(after_key), is_pct)
        rows.append({"Metric": label, "Before Shock": before, "After Shock": after})

    df_comp = pd.DataFrame(rows)
    table = dbc.Table.from_dataframe(df_comp, striped=True, bordered=True, hover=True)

    fig = generate_stress_heatmap(
        E=base["Equity Value"],
        sigma_E=base["Sigma_E"],
        D=base["Debt"],
        r=0.04,
        T=1.0
    )

    return table, fig



if __name__ == '__main__':
    app.run_server(debug=True)
