import dash
from dash import Dash, dcc, html, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import plotly.express as px
import dash_ag_grid as dag
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import requests
import time

# --- Color palette ---
COLOR_DARKBLUE = "#273F4F"
COLOR_MIDBLUE = "#7AA3B9"
COLOR_LIGHTBLUE = "#D1DCE7"
COLOR_ORANGE = "#FE7743"
COLOR_GRAY = "#D7D7D7"
COLOR_BLACK = "#222"
COLOR_DARKGRAY = "#444"

# --- Currency API config ---
EXCHANGE_API_KEY = "12b0d35977f9efec66d264d9"  # <-- Replace with your exchangerate-api.com key
EXCHANGE_BASE_URL = f"https://v6.exchangerate-api.com/v6/{EXCHANGE_API_KEY}/latest/"

order_months = {
    'Month': [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ]
}

# --- AG Grid fejléc stílus (gradient) ---
header_style = {
    "background": "linear-gradient(90deg, #273F4F 0%, #7AA3B9 100%)",
    "color": "#fff",
    "fontWeight": "bold",
    "textAlign": "center",
    "fontSize": "16px",
    "letterSpacing": "1px",
    "borderBottom": f"3px solid {COLOR_MIDBLUE}"
}

# --- Gomb stílusok ---
button_darkgray = {
    "background": COLOR_DARKGRAY,
    "color": "#fff",
    "fontWeight": "bold",
    "border": f"2px solid {COLOR_MIDBLUE}",
    "boxShadow": f"0 2px 8px {COLOR_DARKGRAY}88"
}

# --- Dropdown stílus ---
dropdown_style = {
    "background": COLOR_GRAY,
    "color": COLOR_BLACK,
    "border": f"1px solid {COLOR_DARKBLUE}",
    "fontWeight": "bold"
}

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(
    [
        html.Div(
            style={
                "position": "fixed",
                "top": 0,
                "left": 0,
                "width": "100vw",
                "height": "100vh",
                "zIndex": -1,
                "background": "white",
            }
        ),
        dcc.Markdown(
            "# Stock Market Trading Dashboard",
            style={
                "textAlign": "center",
                "marginTop": "30px",
                "color": COLOR_BLACK,
                "textShadow": "0 2px 8px #fff8"
            }
        ),
        html.Br(),
        html.Br(),

        # --- Currency Calculator Card ---
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader(
                            html.H5("Currency Calculator", style={"textAlign": "center"})
                        ),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col(
                                    dbc.Input(
                                        id="cc-amount",
                                        type="number",
                                        min=0,
                                        step=0.01,
                                        value=1,
                                        placeholder="Amount",
                                        size="lg",
                                        style={"background": "rgba(255,255,255,0.8)"}
                                    ),
                                    width=6,
                                ),
                                dbc.Col(
                                    dcc.Dropdown(
                                        id="cc-from-currency",
                                        options=[
                                            {"label": c, "value": c}
                                            for c in ["USD", "EUR", "HUF", "GBP"]
                                        ],
                                        value="USD",
                                        clearable=False,
                                        style=dropdown_style
                                    ),
                                    width=3,
                                ),
                                dbc.Col(
                                    dcc.Dropdown(
                                        id="cc-to-currency",
                                        options=[
                                            {"label": c, "value": c}
                                            for c in ["USD", "EUR", "HUF", "GBP"]
                                        ],
                                        value="HUF",
                                        clearable=False,
                                        style=dropdown_style
                                    ),
                                    width=3,
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        "Convert",
                                        id="cc-convert-btn",
                                        n_clicks=0,
                                        className="w-100",
                                        style=button_darkgray
                                    ),
                                    width=3,
                                    style={"marginTop": "20px"}
                                ),
                            ], className="mb-2"),
                            html.Br(),
                        ])
                    ],
                    style={
                        "maxWidth": "600px",
                        "margin": "auto",
                        "boxShadow": "0 4px 24px #8ec5fc55, 0 1.5px 8px #e0c3fc55",
                        "borderRadius": "18px",
                        "background": "linear-gradient(120deg, #cceeff 80%, #e0c3fc 100%)"
                    }
                ),
                width=8
            )
        ], justify="center", className="mb-4"),

        # --- Modal for calculator result with spinner ---
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Currency Conversion Result")),
                dbc.ModalBody(
                    dbc.Spinner(html.Div(id="cc-modal-body"))
                ),
                dbc.ModalFooter(
                    dbc.Button("Close", id="cc-modal-close", className="ms-auto", n_clicks=0, style=button_darkgray)
                ),
            ],
            id="cc-modal",
            is_open=False,
        ),
        html.Br(),
        html.Br(),


        # --- Dashboard Controls and Content ---
        dbc.Row([
            dbc.Col(
                dbc.Button("Past Ticker Prices", id="historical-pricing", style=button_darkgray, className="mb-3"),
                width=3
            ),
            dbc.Col(
                dbc.Button("Most Recent Ticker Prices", id="recent-pricing", style=button_darkgray, className="mb-3"),
                width=3
            ),
        ], justify="center"),
        dbc.Alert(id="alert-pricing", duration=5000, is_open=False),
        dbc.Row([
            dbc.Col([dbc.Stack([
                dbc.Label("Timeframe to open new position (business days)"),
                dbc.Input(type="number", min=1, max=10, step=1, value=2, id="business-days")
            ])], width=4),
            dbc.Col([dbc.Stack([
                dbc.Label("Position size (USD)"),
                dbc.Input(type="number", min=50, max=1000, step=50, value=500, id="position-size")
            ])], width=2),
            dbc.Col([dbc.Stack([
                dbc.Label("Ticker setup month"),
                dcc.Dropdown(
                    options=["All"]+order_months['Month'],
                    value='All',
                    clearable=False,
                    id="ticker-setup-month",
                    style=dropdown_style
                )
            ])], width=2),
            dbc.Col([dbc.Stack([
                dbc.Label(" "),
                dbc.Button("Simulate Trading", id="simulate-trading", style=button_darkgray)
            ], gap=4)], width=2),
        ]),
        dbc.Row([], id='cards-row', justify="center", className="my-3"),
        dbc.Row([
            dbc.Col(dbc.Spinner(dcc.Graph(id="profit_n_loss")), width=12),
        ]),
        dbc.Row([
            dbc.Col([dbc.RadioItems(
                id="position-type",
                options=["All", "Long", "Short"],
                value="All",
                inline=True
            )], width=6)
        ]),
        dbc.Row([
            dbc.Col(dbc.Spinner(dcc.Graph(id="trades-open")), width=6),
            dbc.Col(dbc.Spinner(dcc.Graph(id="open-by-action")), width=6)
        ]),
        dbc.Row([
            dbc.Col(dbc.Spinner(dcc.Graph(id="trades-closed")), width=6),
            dbc.Col(dbc.Spinner(dcc.Graph(id="closed-by-action")), width=6)
        ]),
        dbc.Row([
            dbc.Col(html.Div(id="table-space"), width=12),
        ]),
        dcc.Store(id="store-sim-trades")
    ],
    fluid=True,
    style={"paddingBottom": "40px"}
)

# --- Currency Calculator Modal Callback with Spinner ---
@app.callback(
    Output("cc-modal", "is_open"),
    Output("cc-modal-body", "children"),
    [Input("cc-convert-btn", "n_clicks"), Input("cc-modal-close", "n_clicks")],
    [State("cc-amount", "value"),
     State("cc-from-currency", "value"),
     State("cc-to-currency", "value"),
     State("cc-modal", "is_open")],
    prevent_initial_call=True
)
def toggle_cc_modal(n_convert, n_close, amount, from_curr, to_curr, is_open):
    ctx = dash.callback_context
    if not ctx.triggered:
        return is_open, ""
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger_id == "cc-convert-btn":
        if not amount or from_curr == to_curr:
            return True, html.Div(
                "Please enter a valid amount and select different currencies.",
                style={"fontSize": "1.2rem", "fontWeight": "bold", "color": "#FE7743"}
            )
        time.sleep(1)
        url = f"{EXCHANGE_BASE_URL}{from_curr}"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                return True, html.Div("API error!", style={"fontSize": "1.2rem", "fontWeight": "bold", "color": "#FE7743"})
            data = response.json()
            rate = data["conversion_rates"].get(to_curr)
            if not rate:
                return True, html.Div("Currency not supported!", style={"fontSize": "1.2rem", "fontWeight": "bold", "color": "#FE7743"})
            converted = amount * rate
            return True, html.Div([
                html.Div(f"1 {from_curr} = {rate:.4f} {to_curr}",
                         style={"fontSize": "1rem", "color": "#888", "marginBottom": "10px"}),
                html.Div(f"{amount} {from_curr} = {converted:.2f} {to_curr}",
                         style={"fontSize": "2.2rem", "fontWeight": "bold", "color": "#FE7743"})
            ])
        except Exception:
            return True, html.Div("Network or API error!", style={"fontSize": "1.2rem", "fontWeight": "bold", "color": "#FE7743"})
    elif trigger_id == "cc-modal-close":
        return False, ""
    return is_open, ""

# --- Get historical prices for all tickers in setup ticker data from April 1 ---
@callback(
    Output("alert-pricing", "children"),
    Output("alert-pricing", "is_open"),
    Input("historical-pricing", "n_clicks"),
    running=[(Output("historical-pricing", "disabled"), True, False)],
    prevent_initial_call=True
)
def get_past_ticker_prices(_):
    ticker_df = pd.read_csv('trading-setups.csv')
    end_date = datetime.now()
    end_str = end_date.strftime('%Y-%m-%d')
    start_str = '2025-04-01'
    all_data = pd.DataFrame()
    ticker_df.dropna(subset=['ticker'], inplace=True)
    unique_tickers = ticker_df['ticker'].unique()
    for ticker in unique_tickers:
        stock = yf.Ticker(ticker)
        stock_data = stock.history(start=start_str, end=end_str)
        stock_data['Ticker'] = ticker
        all_data = pd.concat([all_data, stock_data])
    all_data = all_data.reset_index()
    all_data.drop(['Dividends', 'Stock Splits'], axis=1, inplace=True)
    all_data.to_csv('ticker-prices.csv', index=False)
    return f"Saved historical ticker prices to ticker-prices.csv", True

# --- Get most recent prices for all tickers ---
@callback(
    Output("alert-pricing", "children", allow_duplicate=True),
    Output("alert-pricing", "is_open", allow_duplicate=True),
    Input("recent-pricing", "n_clicks"),
    running=[(Output("recent-pricing", "disabled"), True, False)],
    prevent_initial_call=True
)
def get_most_recent_ticker_prices(_):
    ticker_df = pd.read_csv('trading-setups.csv')
    unique_tickers = ticker_df['ticker'].unique()
    most_recent_bday = pd.bdate_range(end=pd.Timestamp.today(), periods=1)[0]
    next_day = most_recent_bday + pd.Timedelta(days=1)
    all_data = pd.DataFrame()
    for ticker in unique_tickers:
        stock = yf.Ticker(ticker)
        stock_data = stock.history(start=most_recent_bday, end=next_day)
        stock_data['Ticker'] = ticker
        all_data = pd.concat([all_data, stock_data])
    all_data = all_data.reset_index()
    all_data.drop(['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits', 'Capital Gains'], axis=1, inplace=True)
    all_data.to_csv('ticker-prices-today.csv', index=False)
    return f"Saved most recent ticker prices to ticker-prices-today.csv", True

# --- Simulate Trades ---
@callback(
    Output("table-space", "children"),
    Output("store-sim-trades", "data"),
    Input("simulate-trading", "n_clicks"),
    State("business-days", "value"),
    State("ticker-setup-month", "value"),
    State("position-size", "value"),
    running=[(Output("simulate-trading", "disabled"), True, False)],
    prevent_initial_call=False
)
def trading_simulation(_, business_days, setup_month, position_size):
    trade_setup_df = pd.read_csv('trading-setups.csv')
    trade_setup_df['e_report'] = pd.to_datetime(trade_setup_df['e_report'], format='%m/%d/%Y', errors='coerce').dt.date
    trade_setup_df['observation'] = pd.to_datetime(trade_setup_df['observation'], format='%m/%d/%Y').dt.date
    trade_setup_df['month_name'] = pd.to_datetime(trade_setup_df['observation']).dt.strftime('%B')
    if setup_month == "All":
        trade_setup_df.drop(columns='month_name', inplace=True)
    else:
        trade_setup_df = trade_setup_df[trade_setup_df['month_name'] == setup_month]
        trade_setup_df.drop(columns='month_name', inplace=True)
        trade_setup_df = trade_setup_df.reset_index(drop=True)
    numeric_cols_setup = ['enter_from', 'enter_to', 'stoploss', 'pt1', 'pt2', 'pt3']
    for col in numeric_cols_setup:
        trade_setup_df[col] = pd.to_numeric(trade_setup_df[col], errors='coerce')
    ticker_prices_df = pd.read_csv('ticker-prices.csv')
    ticker_prices_df['Date'] = pd.to_datetime(ticker_prices_df['Date']).dt.date
    numeric_cols_prices = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols_prices:
        ticker_prices_df[col] = pd.to_numeric(ticker_prices_df[col])
    ticker_prices_df.sort_values(by=['Date', 'Ticker'], inplace=True)
    executed_trades_log = []
    open_positions = {}
    unique_dates = sorted(ticker_prices_df['Date'].unique())
    for current_date in unique_dates:
        closed_today_tickers = set()
        daily_prices_for_date = ticker_prices_df[ticker_prices_df['Date'] == current_date]
        tickers_with_open_positions = list(open_positions.keys())
        for ticker in tickers_with_open_positions:
            if ticker not in open_positions:
                continue
            position_details = open_positions[ticker]
            setup_row = trade_setup_df.iloc[position_details['setup_index']]
            ticker_price_data_today = daily_prices_for_date[daily_prices_for_date['Ticker'] == ticker]
            if ticker_price_data_today.empty:
                continue
            current_high_price = ticker_price_data_today['High'].iloc[0]
            current_low_price = ticker_price_data_today['Low'].iloc[0]
            pos_trade_type = position_details['trade_type']
            pos_shares_open = position_details['shares_open']
            current_dynamic_stoploss = position_details['current_stoploss']
            stop_loss_triggered_today = False
            if pos_trade_type == 'short':
                if current_high_price >= current_dynamic_stoploss:
                    executed_trades_log.append({
                        'Date': current_date, 'Ticker': ticker, 'Action': 'Stop-Loss Buy',
                        'Price': current_dynamic_stoploss,
                        'Shares_Traded': pos_shares_open,
                        'Position_Shares_Remaining_After_Trade': 0
                    })
                    del open_positions[ticker]
                    closed_today_tickers.add(ticker)
                    stop_loss_triggered_today = True
            elif pos_trade_type == 'buy':
                if current_low_price <= current_dynamic_stoploss:
                    executed_trades_log.append({
                        'Date': current_date, 'Ticker': ticker, 'Action': 'Stop-Loss Sell',
                        'Price': current_dynamic_stoploss,
                        'Shares_Traded': pos_shares_open,
                        'Position_Shares_Remaining_After_Trade': 0
                    })
                    del open_positions[ticker]
                    closed_today_tickers.add(ticker)
                    stop_loss_triggered_today = True
            if stop_loss_triggered_today:
                continue
            if pos_trade_type == 'short':
                if not position_details['pt1_reached'] and pos_shares_open == 3 and current_low_price <= setup_row['pt1']:
                    executed_trades_log.append({
                        'Date': current_date, 'Ticker': ticker, 'Action': 'PT1 Buy',
                        'Price': setup_row['pt1'], 'Shares_Traded': 1,
                        'Position_Shares_Remaining_After_Trade': 2
                    })
                    position_details['shares_open'] = 2
                    position_details['pt1_reached'] = True
                    position_details['current_stoploss'] = position_details['initial_entry_price']
                    pos_shares_open = 2
                if not position_details['pt2_reached'] and pos_shares_open == 2 and current_low_price <= setup_row['pt2']:
                    executed_trades_log.append({
                        'Date': current_date, 'Ticker': ticker, 'Action': 'PT2 Buy',
                        'Price': setup_row['pt2'], 'Shares_Traded': 1,
                        'Position_Shares_Remaining_After_Trade': 1
                    })
                    position_details['shares_open'] = 1
                    position_details['pt2_reached'] = True
                    position_details['current_stoploss'] = setup_row['pt1']
                    pos_shares_open = 1
                if not position_details['pt3_reached'] and pos_shares_open == 1 and current_low_price <= setup_row['pt3']:
                    executed_trades_log.append({
                        'Date': current_date, 'Ticker': ticker, 'Action': 'PT3 Buy',
                        'Price': setup_row['pt3'], 'Shares_Traded': 1,
                        'Position_Shares_Remaining_After_Trade': 0
                    })
                    del open_positions[ticker]
                    closed_today_tickers.add(ticker)
            elif pos_trade_type == 'buy':
                if not position_details['pt1_reached'] and pos_shares_open == 3 and current_high_price >= setup_row['pt1']:
                    executed_trades_log.append({
                        'Date': current_date, 'Ticker': ticker, 'Action': 'PT1 Sell',
                        'Price': setup_row['pt1'], 'Shares_Traded': 1,
                        'Position_Shares_Remaining_After_Trade': 2
                    })
                    position_details['shares_open'] = 2
                    position_details['pt1_reached'] = True
                    position_details['current_stoploss'] = position_details['initial_entry_price']
                    pos_shares_open = 2
                if not position_details['pt2_reached'] and pos_shares_open == 2 and current_high_price >= setup_row['pt2']:
                    executed_trades_log.append({
                        'Date': current_date, 'Ticker': ticker, 'Action': 'PT2 Sell',
                        'Price': setup_row['pt2'], 'Shares_Traded': 1,
                        'Position_Shares_Remaining_After_Trade': 1
                    })
                    position_details['shares_open'] = 1
                    position_details['pt2_reached'] = True
                    position_details['current_stoploss'] = setup_row['pt1']
                    pos_shares_open = 1
                if not position_details['pt3_reached'] and pos_shares_open == 1 and current_high_price >= setup_row['pt3']:
                    executed_trades_log.append({
                        'Date': current_date, 'Ticker': ticker, 'Action': 'PT3 Sell',
                        'Price': setup_row['pt3'], 'Shares_Traded': 1,
                        'Position_Shares_Remaining_After_Trade': 0
                    })
                    del open_positions[ticker]
                    closed_today_tickers.add(ticker)
        for idx, setup_row in trade_setup_df.iterrows():
            ticker = setup_row['ticker']
            if ticker in closed_today_tickers:
                continue
            if ticker in open_positions:
                continue
            if current_date < setup_row['observation']:
                continue
            observation_timestamp = pd.Timestamp(setup_row['observation'])
            current_timestamp = pd.Timestamp(current_date)
            num_business_days_since_observation = 0
            if current_timestamp > observation_timestamp:
                start_count_timestamp = observation_timestamp + pd.offsets.BDay()
                if start_count_timestamp <= current_timestamp:
                    num_business_days_since_observation = len(
                        pd.bdate_range(start=start_count_timestamp,
                                       end=current_timestamp))
            if num_business_days_since_observation > business_days:
                continue
            ticker_price_data_today = daily_prices_for_date[daily_prices_for_date['Ticker'] == ticker]
            if ticker_price_data_today.empty:
                continue
            current_close_price = ticker_price_data_today['Close'].iloc[0]
            trade_can_be_initiated = False
            actual_entry_price = 0.0
            initial_action_type = ""
            if setup_row['trade'] == 'buy':
                entry_low_bound = setup_row['enter_from']
                entry_high_bound = setup_row['enter_to']
                if entry_low_bound <= current_close_price <= entry_high_bound:
                    actual_entry_price = current_close_price
                    trade_can_be_initiated = True
                    initial_action_type = "Initial Buy"
            elif setup_row['trade'] == 'short':
                entry_low_bound = setup_row['enter_to']
                entry_high_bound = setup_row['enter_from']
                if entry_low_bound <= current_close_price <= entry_high_bound:
                    actual_entry_price = current_close_price
                    trade_can_be_initiated = True
                    initial_action_type = "Initial Short"
            if trade_can_be_initiated:
                executed_trades_log.append({
                    'Date': current_date, 'Ticker': ticker, 'Action': initial_action_type,
                    'Price': actual_entry_price,
                    'Shares_Traded': 3,
                    'Position_Shares_Remaining_After_Trade': 3
                })
                open_positions[ticker] = {
                    'setup_index': idx,
                    'trade_type': setup_row['trade'],
                    'shares_open': 3,
                    'pt1_reached': False, 'pt2_reached': False, 'pt3_reached': False,
                    'initial_entry_price': actual_entry_price,
                    'current_stoploss': setup_row['stoploss']
                }
    executed_trades_df = pd.DataFrame(executed_trades_log)
    if not executed_trades_df.empty:
        executed_trades_df = executed_trades_df[[
            'Date', 'Ticker', 'Action', 'Price',
            'Shares_Traded', 'Position_Shares_Remaining_After_Trade'
        ]]
        executed_trades_df.sort_values(by=['Date', 'Ticker'], inplace=True)
        executed_trades_df.reset_index(drop=True, inplace=True)
    executed_trades_df['Standardized_Multiplier'] = np.nan
    executed_trades_df['Standardized_Trade'] = np.nan
    initial_action_mask = executed_trades_df['Action'].isin(['Initial Buy', 'Initial Short'])
    executed_trades_df.loc[initial_action_mask, 'Standardized_Multiplier'] = position_size / executed_trades_df.loc[initial_action_mask, 'Price']
    executed_trades_df['Standardized_Multiplier'] = executed_trades_df.groupby('Ticker')['Standardized_Multiplier'].ffill()
    for index, row in executed_trades_df.iterrows():
        if pd.isna(row['Standardized_Multiplier']):
            executed_trades_df.loc[index, 'Standardized_Trade'] = np.nan
            continue
        base_standardized_value = row['Standardized_Multiplier'] * row['Price']
        if row['Action'] in ['Initial Buy', 'Initial Short']:
            executed_trades_df.loc[index, 'Standardized_Trade'] = base_standardized_value
        else:
            share_factor = np.nan
            if row['Shares_Traded'] == 1:
                share_factor = 1 / 3
            elif row['Shares_Traded'] == 2:
                share_factor = 2 / 3
            elif row['Shares_Traded'] == 3:
                share_factor = 1.0
            if pd.notna(share_factor):
                executed_trades_df.loc[index, 'Standardized_Trade'] = base_standardized_value * share_factor
            else:
                executed_trades_df.loc[index, 'Standardized_Trade'] = np.nan
    month_names = {
        1: 'January', 2: 'February', 3: 'March', 4: 'April',
        5: 'May', 6: 'June', 7: 'July', 8: 'August',
        9: 'September', 10: 'October', 11: 'November', 12: 'December'
    }
    executed_trades_df['Month'] = pd.to_datetime(executed_trades_df['Date']).dt.month
    executed_trades_df['Month'] = executed_trades_df['Month'].map(month_names)
    buy_actions = ['Initial Buy', 'PT1 Buy', 'PT2 Buy', 'PT3 Buy', 'Stop-Loss Buy']
    sell_actions = ['Initial Short', 'PT1 Sell', 'PT2 Sell', 'PT3 Sell', 'Stop-Loss Sell']
    executed_trades_df['Standardized_Trade'] = executed_trades_df.apply(
        lambda row: row['Standardized_Trade'] * (
            -1 if row['Action'] in buy_actions else 1 if row['Action'] in sell_actions else row['Standardized_Trade']),
        axis=1
    )
    executed_trades_df.to_csv("standardized-executed-trades.csv", index=False)
    grid = dag.AgGrid(
        rowData=executed_trades_df.to_dict("records"),
        columnDefs=[
            {"field": i, 'filter': True, 'sortable': True, "headerStyle": header_style}
            for i in executed_trades_df.columns
        ],
        dashGridOptions={"pagination": True},
        columnSize="sizeToFit"
    )
    return grid, executed_trades_df.to_dict('records')

# --- Build Visualizations ---
@callback(
    Output("trades-open", "figure"),
    Output("open-by-action", "figure"),
    Output("trades-closed", "figure"),
    Output("closed-by-action", "figure"),
    Output("profit_n_loss", "figure"),
    Output("cards-row", "children"),
    Input("store-sim-trades", "data"),
    Input("position-type", "value"),
    State("position-size", "value"),
)
def build_graphs(stored_data, position_type, position_size):
    if stored_data is None:
        return no_update
    df = pd.DataFrame(stored_data)
    positions_opened = df[(df['Action'] == 'Initial Buy') | (df['Action'] == 'Initial Short')]
    trades_count = positions_opened.groupby('Month').size().reset_index(name='Trade Count')
    fig_trades_month = px.bar(
        trades_count, x='Month', y='Trade Count', title='Positions Opened by Month',
        category_orders=order_months,
        color_discrete_sequence=[COLOR_DARKBLUE, COLOR_MIDBLUE, COLOR_LIGHTBLUE]
    )
    trades_count = positions_opened.groupby(['Month', 'Action']).size().reset_index(name='Trade Count')
    fig_trades_action = px.histogram(
        trades_count, x='Month', y='Trade Count', color='Action',
        barmode='group', title='Positions Opened by Month AND Trade',
        category_orders=order_months,
        color_discrete_sequence=[COLOR_DARKBLUE, COLOR_MIDBLUE, COLOR_LIGHTBLUE]
    )
    closed_trades_df = df[df['Position_Shares_Remaining_After_Trade'] == 0]
    closed_trades_count = closed_trades_df.groupby('Month').size().reset_index(name='Trade Count')
    fig_closed = px.bar(
        closed_trades_count, x='Month', y='Trade Count', title='Positions Closed by Month',
        category_orders=order_months,
        color_discrete_sequence=[COLOR_DARKBLUE, COLOR_MIDBLUE, COLOR_LIGHTBLUE]
    )
    trades_count_action = closed_trades_df.groupby(['Month', 'Action']).size().reset_index(name='Trade Count')
    fig_closed_trades_action = px.histogram(
        trades_count_action, x='Month', y='Trade Count', color='Action',
        barmode='group', title='Positions Closed by Trade',
        category_orders=order_months,
        color_discrete_sequence=[COLOR_DARKBLUE, COLOR_MIDBLUE, COLOR_LIGHTBLUE]
    )
    if position_type == 'Long':
        position_chosen_df = df[df['Action'] == 'Initial Buy']
    elif position_type == 'Short':
        position_chosen_df = df[df['Action'] == 'Initial Short']
    else:
        position_chosen_df = df[(df['Action'] == 'Initial Buy') | (df['Action'] == 'Initial Short')]
    current_prices_df = pd.read_csv('ticker-prices-today.csv')
    current_prices_df['Date'] = pd.to_datetime(current_prices_df['Date']).dt.date
    price_lookup_series = current_prices_df.set_index('Ticker')['Close']
    closed_trades_tickers = position_chosen_df[position_chosen_df['Position_Shares_Remaining_After_Trade'] == 0]['Ticker'].unique()
    pnl_summary = []
    for ticker, group in df.groupby('Ticker'):
        group = group.sort_values(by='Date')
        if position_type == 'Long':
            initial_trades = group[group['Action'].str.contains('Initial Buy')]
        elif position_type == 'Short':
            initial_trades = group[group['Action'].str.contains('Initial Short')]
        else:
            initial_trades = group[group['Action'].str.contains('Initial')]
        if initial_trades.empty:
            continue
        total_initial_shares = initial_trades['Shares_Traded'].sum()
        total_initial_std_value = initial_trades['Standardized_Trade'].sum()
        avg_entry_std_price_per_share = total_initial_std_value / total_initial_shares
        is_short = 'Short' in initial_trades.iloc[0]['Action']
        realized_pnl = 0
        closing_trades = group[~group['Action'].str.contains('Initial')]
        for _, row in closing_trades.iterrows():
            shares_closed = row['Shares_Traded']
            exit_std_price_per_share = abs(row['Standardized_Trade']) / shares_closed
            if is_short:
                pnl_per_share = avg_entry_std_price_per_share - exit_std_price_per_share
            else:
                pnl_per_share = exit_std_price_per_share + avg_entry_std_price_per_share
            realized_pnl += pnl_per_share * shares_closed
        unrealized_pnl = 0
        shares_remaining = group.iloc[-1]['Position_Shares_Remaining_After_Trade']
        if shares_remaining > 0:
            current_market_price = price_lookup_series.get(ticker)
            if current_market_price is not None:
                initial_anchor_price = initial_trades.iloc[0]['Price']
                current_std_price_per_share = (current_market_price / initial_anchor_price) * avg_entry_std_price_per_share
                if is_short:
                    unrealized_pnl_per_share = avg_entry_std_price_per_share - current_std_price_per_share
                    unrealized_pnl = unrealized_pnl_per_share * shares_remaining
                else:
                    unrealized_pnl_per_share = current_std_price_per_share - avg_entry_std_price_per_share
                    unrealized_pnl = unrealized_pnl_per_share * -shares_remaining
        pnl_summary.append({
            'Ticker': ticker,
            'Status': 'Open' if shares_remaining > 0 else 'Closed',
            'Shares Open': shares_remaining,
            'Realized P&L($)': realized_pnl,
            'Unrealized P&L($)': unrealized_pnl,
            'Total P&L($)': realized_pnl + unrealized_pnl
        })
    summary_df = pd.DataFrame(pnl_summary)
    total_realized_pnl = summary_df['Realized P&L($)'].sum()
    total_unrealized_pnl = summary_df['Unrealized P&L($)'].sum()
    total_pnl = summary_df['Total P&L($)'].sum()
    total_capital_deployed = len(position_chosen_df) * position_size

    # Kártyák színezése
    c_cap_deplyd = dbc.Card(
        [
            dbc.CardBody(
                [
                    html.H5("Capital Deployed", className="card-title"),
                    html.Hr(),
                    html.P(f"${total_capital_deployed}", className="card-text"),
                ]
            ),
        ],
        style={"background": COLOR_DARKBLUE, "color": "#fff", "border": f"2px solid {COLOR_ORANGE}"}
    )
    c_unrlzd_prft = dbc.Card(
        [
            dbc.CardBody(
                [
                    html.H5("Unrealized P&L", className="card-title"),
                    html.Hr(),
                    html.P(f"${total_unrealized_pnl:.2f}", className="card-text"),
                ]
            ),
        ],
        style={"background": COLOR_MIDBLUE, "color": "#fff", "border": f"2px solid {COLOR_ORANGE}"}
    )
    c_rlzd_prft = dbc.Card(
        [
            dbc.CardBody(
                [
                    html.H5("Realized P&L", className="card-title"),
                    html.Hr(),
                    html.P(f"${total_realized_pnl:.2f}", className="card-text"),
                ]
            ),
        ],
        style={"background": COLOR_LIGHTBLUE, "color": "#222", "border": f"2px solid {COLOR_DARKBLUE}"}
    )
    c_proft_dlr = dbc.Card(
        [
            dbc.CardBody(
                [
                    html.H5("Total Net P&L", className="card-title"),
                    html.Hr(),
                    html.P(f"${total_pnl:.2f}", className="card-text"),
                ]
            ),
        ],
        style={"background": COLOR_BLACK, "color": "#fff", "border": f"2px solid {COLOR_ORANGE}"}
    )
    c_proft_pct = dbc.Card(
        [
            dbc.CardBody(
                [
                    html.H5("Total Net P&L (%)", className="card-title"),
                    html.Hr(),
                    html.P(f"{(total_pnl / total_capital_deployed) * 100:.2f}%", className="card-text"),
                ]
            ),
        ],
        style={"background": COLOR_DARKBLUE, "color": "#fff", "border": f"2px solid {COLOR_ORANGE}"}
    )
    cards = [
        dbc.Col(c_cap_deplyd, width=2),
        dbc.Col(c_unrlzd_prft, width=2),
        dbc.Col(c_rlzd_prft, width=2),
        dbc.Col(c_proft_dlr, width=2),
        dbc.Col(c_proft_pct, width=2),
    ]
    fig_pnl = px.bar(
        summary_df, x='Ticker', y=['Realized P&L($)', 'Unrealized P&L($)'],
        color_discrete_sequence=[COLOR_DARKBLUE, COLOR_MIDBLUE, COLOR_LIGHTBLUE]
    )
    fig_pnl.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    return fig_trades_month, fig_trades_action, fig_closed, fig_closed_trades_action, fig_pnl, cards

if __name__ == '__main__':
    app.run(debug=True)