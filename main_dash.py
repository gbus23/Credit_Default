import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import yfinance as yf

from data_loader import fetch_stock_data, get_fundamentals, prime_cache_with_history
from merton_model import solve_merton_model
from Black_Cox_model import black_cox_pd
from rolling_analysis import historical_quarterly_pd_series
from stress_test import stress_test


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server


# ---------- Helpers ----------

def latest_debt_asof_date(ticker: str):
    """
    Return the most recent balance-sheet column date (as-of) for debt snapshot.
    """
    try:
        bs = yf.Ticker(ticker).balance_sheet
        if bs is not None and not bs.empty:
            return pd.to_datetime(bs.columns[0]).date()
    except Exception:
        pass
    return None


def run_analysis(ticker: str, r: float = 0.04, mode: str = "merton"):
    """
    One-shot analysis for current PD. Uses *current* equity inputs (price & σE)
    and latest available *total debt* snapshot.
    """
    price, sigma_E = fetch_stock_data(ticker)
    # get_fundamentals should return: shares_outstanding, total_debt, kmv_debt(dummy), short_term, long_term
    shares_outstanding, total_debt, short_term_debt, long_term_debt, debt_asof, debt_source = get_fundamentals(ticker)


    T = 1.0
    E = price * shares_outstanding

    if mode == "black_cox":
        result = black_cox_pd(E, sigma_E, D=total_debt, B=total_debt, r=r, T=T)
        PD = result["PD_BC"]
    else:  # merton
        result = solve_merton_model(E, sigma_E, D=total_debt, r=r, T=T)
        PD = result["PD"]

    output = {
        "Ticker": ticker,
        "Price": price,
        "Shares Outstanding": shares_outstanding,
        "Equity Value": E,
        "Total Debt": total_debt,
        "Short-Term Debt": short_term_debt,
        "Long-Term Debt": long_term_debt,
        "Debt (used in model)": total_debt,
        "Sigma_E": sigma_E,
        "V": result.get("V"),
        "Sigma_V": result.get("sigma_V"),
        "PD": PD,
        "DistanceToDefault": result.get("DistanceToDefault", None),
        "Maturity (T)": T,
        "Mode": mode
    }

    return pd.DataFrame.from_dict(output, orient='index', columns=['Value'])


def format_pd_dynamic(value):
    if value > 0.01:
        return f"{value:.2%}"
    elif value > 1e-4:
        return f"{value:.5%}"
    else:
        return f"{value:.1e}"


def get_rolling_plot(ticker: str, mode: str, current_pd: float | None = None, debt_asof=None):
    """
    Build rolling PD figure from quarterly series and append a 'Today' point equal
    to the current PD (ensuring plot's last point == summary table PD).
    A hover note indicates the debt as-of date used for the 'Today' point.
    """
    prime_cache_with_history(ticker, years=5)

    pd_series = historical_quarterly_pd_series(ticker, mode=mode)

    if isinstance(pd_series, pd.DataFrame) and 'PD' in pd_series.columns:
        pd_series = pd_series["PD"]

    pd_series = pd_series.replace([np.inf, -np.inf], np.nan).dropna()
    pd_series = pd_series[~pd_series.index.duplicated(keep='first')]
    pd_series.index = pd.to_datetime(pd_series.index, errors='coerce')
    pd_series = pd_series[~pd_series.index.isna()]
    pd_series = pd_series.sort_index()

    # ---- Append 'Today' point from current_pd (Option B) ----
    today = pd.Timestamp.today().normalize()
    if current_pd is not None:
        pd_series.loc[today] = float(current_pd)
        pd_series = pd_series.sort_index()

    if pd_series.empty:
        return go.Figure(layout={
            "title": f"No data available for {ticker.upper()} ({mode.upper()})",
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
        })

    # Hover text
    hovertext = []
    for dt, val in pd_series.items():
        if current_pd is not None and dt.normalize() == today:
            if debt_asof:
                hovertext.append(
                    f"Today<br>PD: {format_pd_dynamic(val)}<br>"
                    f"<span style='font-size:11px'>Debt as of {debt_asof}</span>"
                )
            else:
                hovertext.append(f"Today<br>PD: {format_pd_dynamic(val)}")
        else:
            hovertext.append(f"{dt:%b %Y}<br>PD: {format_pd_dynamic(val)}")

    tickvals = sorted(set(pd_series.values))
    ticktext = [format_pd_dynamic(val) for val in tickvals]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd_series.index,
        y=pd_series.values,
        mode='lines+markers',
        name='PD',
        line=dict(width=2),
        hoverinfo='text',
        text=hovertext
    ))

    fig.update_layout(
        title=dict(text=f"{ticker.upper()}  Rolling Probability of Default ({mode.upper()})", x=0.5),
        xaxis_title="Date",
        yaxis_title="Probability of Default",
        xaxis=dict(
            tickformat="%Y-%m",
            tickangle=45,
            showgrid=True,
            gridcolor='lightgrey',
            ticks="outside"
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgrey',
            tickvals=tickvals,
            ticktext=ticktext,
            ticks="outside"
        ),
        template="plotly_white",
        height=500,
        width=800,
        margin=dict(l=60, r=20, t=60, b=60)
    )

    return fig


def generate_stress_heatmap(E, sigma_E, D, r, T=1.0):
    sigma_shocks = np.arange(-50, 105, 5)
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


# ---------- Layout ----------

app.layout = dbc.Container([
    html.H1("Merton / Black–Cox Credit Risk Models", className="text-center my-4"),

    dbc.Row([
        dbc.Col([
            dbc.Input(id="ticker", placeholder="Enter ticker (e.g. AAL)", type="text", className="mb-2"),
            dcc.Dropdown(
                id="mode",
                options=[
                    {"label": "Merton Model", "value": "merton"},
                    {"label": "Black–Cox Model", "value": "black_cox"}
                ],
                placeholder="Select model",
                className="mb-3"
            ),
            dbc.Button("Run Analysis", id="run_button", color="primary", className="w-100")
        ], width=6)
    ], justify="center", className="mb-4"),

    dbc.Row([
        dbc.Col(html.Div(id="company_description"), width=8)
    ], justify="center", className="mb-4"),

    dbc.Row([
        dbc.Col([
            html.H5("Financial Summary", className="text-center mb-3"),
            dbc.Card([dbc.CardBody(html.Div(id="output_table"))], className="mb-4")
        ])
    ], justify="center"),

    dbc.Row([
        dbc.Col(dcc.Loading(dcc.Graph(id="rolling_plot")))
    ], justify="center"),

    # Warning about last point ("Today") mixing current equity with last debt snapshot
    dbc.Row([
        dbc.Col(html.Div(id="pd_warning"), width=8)
    ], justify="center", className="mb-2"),

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

    dbc.Row([
        dbc.Col(html.Div(id="stress_table"), width=6)
    ], justify="center", className="mb-4"),

    dbc.Row([
        dbc.Col(dcc.Loading(dcc.Graph(id="stress_heatmap")), width=8)
    ], justify="center", className="mb-4")
], fluid=True)


# ---------- Callbacks ----------

@app.callback(
    Output("output_table", "children"),
    Output("rolling_plot", "figure"),
    Output("company_description", "children"),
    Output("pd_warning", "children"),
    Input("run_button", "n_clicks"),
    State("ticker", "value"),
    State("mode", "value")
)
def update_rolling_output(n_clicks, ticker, mode):
    if not n_clicks or not ticker or not mode:
        return None, go.Figure(), None, None

    df = run_analysis(ticker, mode=mode)
    raw = df.to_dict()["Value"]

    # Pretty labels
    clean_labels = {
        "Ticker": "Ticker",
        "Price": "Price",
        "Shares Outstanding": "Shares Outstanding",
        "Equity Value": "Equity Value",
        "Total Debt": "Total Debt",
        "Short-Term Debt": "Short-Term Debt",
        "Long-Term Debt": "Long-Term Debt",
        "Debt (used in model)": "Debt (Used in Model)",
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
            elif label in {"Equity Value", "Firm Value (V)", "Price"}:
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

    description = yf.Ticker(ticker).info
    name = description.get("longName", ticker)
    sector = description.get("sector", "N/A")
    industry = description.get("industry", "N/A")
    employees = description.get("fullTimeEmployees", None)
    market_cap = description.get("marketCap", None)
    text = f"{name} operates in the {sector.lower()} sector, specifically in the {industry.lower()} industry."
    if employees:
        text += f" It employs approximately {employees:,} people"
    if market_cap:
        mc = f'${market_cap/1e9:.1f}B' if market_cap and market_cap > 1e9 else f'${(market_cap or 0)/1e6:.1f}M'
        text += f" and has a market capitalization of about {mc}."
    description_card = dbc.Card([dbc.CardHeader("Company Overview"), dbc.CardBody(html.P(text, className="card-text"))])

    # ---- Rolling figure with 'Today' point appended ----
    debt_asof = latest_debt_asof_date(ticker)
    fig = get_rolling_plot(ticker, mode, current_pd=raw["PD"], debt_asof=debt_asof)

    # ---- Warning alert ----
    warn = None
    if debt_asof:
        warn = dbc.Alert(
            f"Note: The last point ('Today') uses current equity (price & volatility) with debt as of {debt_asof} (latest quarterly snapshot).",
            color="warning",
            dismissable=True,
            is_open=True,
            className="mt-2"
        )

    return table, fig, description_card, warn


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
        D=base["Debt (used in model)"],
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
        "Debt (Used in Model)": ("Debt (used in model)", "D_shock"),
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
        D=base["Debt (used in model)"],
        r=0.04,
        T=1.0
    )

    return table, fig


if __name__ == '__main__':
    app.run_server(debug=True)
