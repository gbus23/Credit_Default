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

app.layout = dbc.Container([
    html.H1("Merton / KMV Credit Risk Model", className="text-center my-4"),

    # Inputs centrés
    dbc.Row([
        dbc.Col([
            dbc.Input(id="ticker", placeholder="Enter ticker (e.g. AAPL)", type="text", className="mb-2"),
            dcc.Dropdown(id="analysis_type", options=[
                {"label": "Spot Analysis", "value": "spot"},
                {"label": "Rolling Analysis", "value": "rolling"}
            ], placeholder="Select analysis type", className="mb-2"),
            dcc.Dropdown(id="mode", options=[
                {"label": "Merton Model", "value": "merton"},
                {"label": "KMV Model", "value": "kmv"}
            ], placeholder="Select model", className="mb-3"),
            dbc.Button("Run Analysis", id="run_button", color="primary", className="w-100")
        ], width=6)
    ], justify="center", className="mb-4"),

    # Company Overview centré
    dbc.Row([
        dbc.Col([
            html.Div(id="company_description")
        ], width=8)
    ], justify="center", className="mb-5"),

    # Résultats : graph + tableau
    dbc.Row([
        dbc.Col([
            dcc.Loading(dcc.Graph(id="rolling_plot"))
        ], width=8),

        dbc.Col([
            html.H5("Financial Summary", className="mb-3"),
            html.Div(id="output_table")
        ], width=4)
    ])
], fluid=True)

@app.callback(
    Output("output_table", "children"),
    Output("rolling_plot", "figure"),
    Output("company_description", "children"),
    Input("run_button", "n_clicks"),
    State("ticker", "value"),
    State("analysis_type", "value"),
    State("mode", "value")
)
def update_output(n_clicks, ticker, analysis_type, mode):
    if not n_clicks or not ticker or not analysis_type or not mode:
        return None, go.Figure(), None

    df = run_analysis(ticker, mode=mode)
    table = dbc.Table.from_dataframe(df.reset_index().rename(columns={"index": "Metric"}), striped=True, bordered=True, hover=True)

    description = get_company_summary(ticker)
    description_card = dbc.Card([
        dbc.CardHeader("Company Overview"),
        dbc.CardBody(html.P(description, className="card-text"))
    ])

    if analysis_type == "rolling":
        fig = get_rolling_plot(ticker, mode)
    else:
        fig = go.Figure()

    return table, fig, description_card

if __name__ == '__main__':
    app.run_server(debug=True)
