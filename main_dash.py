import dash
from dash import dcc, html, Input, Output, State, ctx, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import numpy as np

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

def format_quarter_label(date_index):
    return [f"Q{((d.month - 1) // 3) + 1} {d.year}" for d in date_index]

 


def get_rolling_plot(ticker: str, mode: str):
    prime_cache_with_history(ticker, years=5)
    
    pd_series = historical_quarterly_pd_series(ticker, mode=mode)
    # Nettoyage : on enlève les index NaT ou NaN
    pd_series = pd_series[~pd_series.index.isna()]
    pd_series = pd_series.replace([np.inf, -np.inf], np.nan).dropna()
    pd_series = pd_series[~pd_series.index.duplicated(keep='first')]
    print(pd_series)
    pd_series = pd_series.sort_index()


    if pd_series.empty:
        return go.Figure(layout={
            "title": f"Aucune donnée disponible pour {ticker.upper()} ({mode.upper()})",
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
        })

    fig = go.Figure()
    df = pd_series.reset_index()
    df.columns = ["Date", "PD"]

    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["PD"],
        mode='lines+markers',
        name='PD',
        line=dict(width=2)
    ))

    fig.update_layout(
        title=f"Rolling Default Probability - {ticker.upper()} ({mode.upper()})",
        xaxis_title="Date",
        yaxis_title="Default Probability",
        template="plotly_white",
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True, tickformat=".2%")
    )
    return fig
app.layout = dbc.Container([
    html.H1("Merton / KMV Credit Risk Model"),
    dbc.Row([
        dbc.Col([
            dbc.Input(id="ticker", placeholder="Enter ticker (e.g. AAPL)", type="text"),
            dcc.Dropdown(id="analysis_type", options=[
                {"label": "Spot Analysis", "value": "spot"},
                {"label": "Rolling Analysis", "value": "rolling"}
            ], placeholder="Select analysis type"),
            dcc.Dropdown(id="mode", options=[
                {"label": "Merton Model", "value": "merton"},
                {"label": "KMV Model", "value": "kmv"}
            ], placeholder="Select model"),
            html.Br(),
            dbc.Button("Run Analysis", id="run_button", color="primary", className="w-100")
        ], width=4),

        dbc.Col([
            html.Div(id="output_table"),
            dcc.Loading(dcc.Graph(id="rolling_plot"))
        ], width=8)
    ])
], fluid=True)

@app.callback(
    Output("output_table", "children"),
    Output("rolling_plot", "figure"),
    Input("run_button", "n_clicks"),
    State("ticker", "value"),
    State("analysis_type", "value"),
    State("mode", "value")
)
def update_output(n_clicks, ticker, analysis_type, mode):
    if not n_clicks or not ticker or not analysis_type or not mode:
        return None, go.Figure()

    if analysis_type == "rolling":
        fig = get_rolling_plot(ticker, mode)
        return None, fig
    else:
        df = run_analysis(ticker, mode=mode)
        table = dbc.Table.from_dataframe(df.reset_index().rename(columns={"index": "Metric"}), striped=True, bordered=True, hover=True)
        return table, go.Figure()

if __name__ == '__main__':
    app.run_server(debug=True)
