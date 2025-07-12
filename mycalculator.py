import dash
from dash import dcc, html, dash_table, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
from datetime import datetime

# --- Download prices from Yahoo Finance ---
def download_prices(trading_data_path='trading-data.csv', start_str='2025-04-01'):
    end_date = datetime.now()
    end_str = end_date.strftime('%Y-%m-%d')
    ticker_df = pd.read_csv(trading_data_path)
    unique_tickers = ticker_df['ticker'].unique()
    all_data = pd.DataFrame()
    for ticker in unique_tickers:
        stock = yf.Ticker(ticker)
        stock_data = stock.history(start=start_str, end=end_str)
        if stock_data.empty:
            continue
        stock_data['Ticker'] = ticker
        all_data = pd.concat([all_data, stock_data])
    if all_data.empty:
        return None
    all_data = all_data.reset_index()
    all_data.drop(['Dividends', 'Stock Splits'], axis=1, inplace=True, errors='ignore')
    return all_data

# --- Simulate trades ---
def simulate_trades(trading_data_path='trading-data.csv', prices_df=None):
    trade_setup_df = pd.read_csv(trading_data_path)
    trade_setup_df['observation'] = pd.to_datetime(
        trade_setup_df['observation'], format='%m/%d/%Y'
    ).dt.date
    trade_setup_df['e_report'] = pd.to_datetime(
        trade_setup_df['e_report'], format='%m/%d/%Y', errors='coerce'
    ).dt.date
    numeric_cols_setup = [
        'enter_from', 'enter_to', 'stoploss', 'pt1', 'pt2', 'pt3', 'pt4'
    ]
    for col in numeric_cols_setup:
        trade_setup_df[col] = pd.to_numeric(trade_setup_df[col], errors='coerce')
    ticker_prices_df = prices_df.copy()
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
        daily_prices_for_date = ticker_prices_df[
            ticker_prices_df['Date'] == current_date
        ]
        tickers_with_open_positions = list(open_positions.keys())
        for ticker in tickers_with_open_positions:
            if ticker not in open_positions:
                continue
            position_details = open_positions[ticker]
            setup_row = trade_setup_df.iloc[position_details['setup_index']]
            ticker_price_data_today = daily_prices_for_date[
                daily_prices_for_date['Ticker'] == ticker
            ]
            if ticker_price_data_today.empty:
                continue
            current_high_price = ticker_price_data_today['High'].iloc[0]
            current_low_price = ticker_price_data_today['Low'].iloc[0]
            pos_trade_type = position_details['trade_type']
            pos_shares_open = position_details['shares_open']
            stop_loss_triggered_today = False
            if pos_trade_type == 'short':
                if current_high_price >= setup_row['stoploss']:
                    executed_trades_log.append({
                        'Date': current_date, 'Ticker': ticker, 'Action': 'Stop-Loss Buy',
                        'Price': setup_row['stoploss'],
                        'Shares_Traded': pos_shares_open,
                        'Position_Shares_Remaining_After_Trade': 0
                    })
                    del open_positions[ticker]
                    closed_today_tickers.add(ticker)
                    stop_loss_triggered_today = True
            elif pos_trade_type == 'buy':
                if current_low_price <= setup_row['stoploss']:
                    executed_trades_log.append({
                        'Date': current_date, 'Ticker': ticker, 'Action': 'Stop-Loss Sell',
                        'Price': setup_row['stoploss'],
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
                    pos_shares_open = 2
                if not position_details['pt2_reached'] and pos_shares_open == 2 and current_low_price <= setup_row['pt2']:
                    executed_trades_log.append({
                        'Date': current_date, 'Ticker': ticker, 'Action': 'PT2 Buy',
                        'Price': setup_row['pt2'], 'Shares_Traded': 1,
                        'Position_Shares_Remaining_After_Trade': 1
                    })
                    position_details['shares_open'] = 1
                    position_details['pt2_reached'] = True
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
                    pos_shares_open = 2
                if not position_details['pt2_reached'] and pos_shares_open == 2 and current_high_price >= setup_row['pt2']:
                    executed_trades_log.append({
                        'Date': current_date, 'Ticker': ticker, 'Action': 'PT2 Sell',
                        'Price': setup_row['pt2'], 'Shares_Traded': 1,
                        'Position_Shares_Remaining_After_Trade': 1
                    })
                    position_details['shares_open'] = 1
                    position_details['pt2_reached'] = True
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
            if current_date <= setup_row['observation']:
                continue
            ticker_price_data_today = daily_prices_for_date[
                daily_prices_for_date['Ticker'] == ticker
            ]
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
                    'entry_price': actual_entry_price
                }
    executed_trades_df = pd.DataFrame(executed_trades_log)
    if not executed_trades_df.empty:
        executed_trades_df = executed_trades_df[[
            'Date', 'Ticker', 'Action', 'Price',
            'Shares_Traded', 'Position_Shares_Remaining_After_Trade'
        ]]
        executed_trades_df.sort_values(by=['Date', 'Ticker'], inplace=True)
        executed_trades_df.reset_index(drop=True, inplace=True)
    return executed_trades_df

def standardize_executed_trades(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Standardized_Multiplier'] = np.nan
    df['Standardized_Trade'] = np.nan
    initial_action_mask = df['Action'].isin(['Initial Buy', 'Initial Short'])
    df.loc[initial_action_mask, 'Standardized_Multiplier'] = (
        50 / df.loc[initial_action_mask, 'Price']
    )
    df['Standardized_Multiplier'] = df.groupby('Ticker')['Standardized_Multiplier'].ffill()
    for index, row in df.iterrows():
        if pd.isna(row['Standardized_Multiplier']):
            df.loc[index, 'Standardized_Trade'] = np.nan
            continue
        base_standardized_value = row['Standardized_Multiplier'] * row['Price']
        if row['Action'] in ['Initial Buy', 'Initial Short']:
            df.loc[index, 'Standardized_Trade'] = base_standardized_value
        else:
            share_factor = np.nan
            if row['Shares_Traded'] == 1:
                share_factor = 1/3
            elif row['Shares_Traded'] == 2:
                share_factor = 2/3
            elif row['Shares_Traded'] == 3:
                share_factor = 1.0
            if pd.notna(share_factor):
                df.loc[index, 'Standardized_Trade'] = base_standardized_value * share_factor
            else:
                df.loc[index, 'Standardized_Trade'] = np.nan
    month_names = {
        1: 'January', 2: 'February', 3: 'March', 4: 'April',
        5: 'May', 6: 'June', 7: 'July', 8: 'August',
        9: 'September', 10: 'October', 11: 'November', 12: 'December'
    }
    df['Month'] = df['Date'].dt.month.map(month_names)
    buy_actions = ['Initial Buy', 'PT1 Buy', 'PT2 Buy', 'PT3 Buy', 'Stop-Loss Buy']
    sell_actions = ['Initial Short', 'PT1 Sell', 'PT2 Sell', 'PT3 Sell', 'Stop-Loss Sell']
    def sign_trade(row):
        if row['Action'] in buy_actions:
            return -abs(row['Standardized_Trade'])
        elif row['Action'] in sell_actions:
            return abs(row['Standardized_Trade'])
        else:
            return row['Standardized_Trade']
    df['Standardized_Trade'] = df.apply(sign_trade, axis=1)
    return df

def analyze_outcomes(df, month, ticker=None):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df_full = df.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)
    if ticker and ticker != 'ALL':
        df_full = df_full[df_full['Ticker'] == ticker]
    trades = df_full[df_full['Month'] == month].copy()
    target_actions = ['Initial Buy', 'Initial Short']
    filtered_actions = trades[trades['Action'].isin(target_actions)]
    action_counts = filtered_actions.groupby(['Ticker', 'Action']).size()
    result_df = action_counts.unstack(fill_value=0)
    if 'Initial Buy' not in result_df.columns:
        result_df['Initial Buy'] = 0
    if 'Initial Short' not in result_df.columns:
        result_df['Initial Short'] = 0
    result_df = result_df[['Initial Buy', 'Initial Short']]
    tickers_from_original_result_df = result_df.reset_index()['Ticker'].unique()
    initial_positions_to_analyze = df_full[
        (df_full['Month'] == month) &
        (df_full['Action'].isin(['Initial Buy', 'Initial Short'])) &
        (df_full['Ticker'].isin(tickers_from_original_result_df))
    ].copy()
    outcome_details_list = []
    for _, initial_trade_row in initial_positions_to_analyze.iterrows():
        current_ticker = initial_trade_row['Ticker']
        initial_action_type = initial_trade_row['Action']
        initial_trade_original_index = initial_trade_row.name
        subsequent_trades_for_ticker_all  = df_full[
            (df_full['Ticker'] == current_ticker) &
            (df_full.index > initial_trade_original_index)
        ]
        next_opening_trade_df_index = -1
        for idx_in_df_full, trade in subsequent_trades_for_ticker_all.iterrows():
            if trade['Action'] in ['Initial Buy', 'Initial Short']:
                next_opening_trade_df_index = idx_in_df_full
                break
        if next_opening_trade_df_index != -1:
            relevant_subsequent_trades = subsequent_trades_for_ticker_all[
                subsequent_trades_for_ticker_all.index < next_opening_trade_df_index
            ]
        else:
            relevant_subsequent_trades = subsequent_trades_for_ticker_all
        actions_list = relevant_subsequent_trades['Action'].tolist()
        outcome = 'unknown'
        outcome_dollar = 0
        realized = False
        if initial_action_type == 'Initial Buy':
            if len(actions_list) >= 1:
                action1 = actions_list[0]
                if action1 == 'Stop-Loss Sell':
                    outcome = 'failed'
                    outcome_dollar = -27
                    realized = True
                elif action1 == 'PT1 Sell':
                    if len(actions_list) == 2:
                        action2 = actions_list[1]
                        if action2 == 'Stop-Loss Sell':
                            outcome = 'succeeded'
                            outcome_dollar = 13
                            realized = True
                        elif action2 == 'PT2 Sell':
                            outcome = 'succeeded'
                            outcome_dollar = 36
                            realized = False
                    elif len(actions_list) == 3:
                        action3 = actions_list[2]
                        if action3 == 'Stop-Loss Sell':
                            outcome = 'succeeded'
                            outcome_dollar = 49
                            realized = True
                        elif action3 == 'PT3 Sell':
                            outcome = 'succeeded'
                            outcome_dollar = 74
                            realized = True
                    else:
                        outcome = 'succeeded'
                        outcome_dollar = 13
                        realized = False
        elif initial_action_type == 'Initial Short':
            if len(actions_list) >= 1:
                action1 = actions_list[0]
                if action1 == 'Stop-Loss Buy':
                    outcome = 'failed'
                    outcome_dollar = -27
                    realized = True
                elif action1 == 'PT1 Buy':
                    if len(actions_list) == 2:
                        action2 = actions_list[1]
                        if action2 == 'Stop-Loss Buy':
                            outcome = 'succeeded'
                            outcome_dollar = 13
                            realized = True
                        elif action2 == 'PT2 Buy':
                            outcome = 'succeeded'
                            outcome_dollar = 36
                            realized = False
                    elif len(actions_list) == 3:
                        action3 = actions_list[2]
                        if action3 == 'Stop-Loss Buy':
                            outcome = 'succeeded'
                            outcome_dollar = 49
                            realized = True
                        elif action3 == 'PT3 Buy':
                            outcome = 'succeeded'
                            outcome_dollar = 74
                            realized = True
                    else:
                        outcome = 'succeeded'
                        outcome_dollar = 13
                        realized = False
        outcome_details_list.append({
            'Date_of_Initial_Action': initial_trade_row['Date'],
            'Ticker': initial_trade_row['Ticker'],
            'Initial_Action_Type': initial_action_type,
            'Initial_Price': initial_trade_row['Price'],
            'Outcome': outcome,
            'Outcome_dollar': outcome_dollar,
            'Realized': realized
        })
    final_outcomes_df = pd.DataFrame(outcome_details_list)
    outcome_counts = final_outcomes_df['Outcome'].value_counts()
    all_possible_outcomes = ['failed', 'succeeded', 'unknown']
    outcome_counts = outcome_counts.reindex(all_possible_outcomes, fill_value=0)
    total_outcome = final_outcomes_df['Outcome_dollar'].sum()
    realized_outcome = final_outcomes_df[final_outcomes_df['Realized']]['Outcome_dollar'].sum()
    unrealized_outcome = final_outcomes_df[~final_outcomes_df['Realized']]['Outcome_dollar'].sum()
    final_outcomes_df_short = final_outcomes_df[
        final_outcomes_df['Initial_Action_Type'] == 'Initial Short'
    ]
    final_outcomes_df_long = final_outcomes_df[
        final_outcomes_df['Initial_Action_Type'] == 'Initial Buy'
    ]
    short_outcome = final_outcomes_df_short['Outcome_dollar'].sum()
    long_outcome = final_outcomes_df_long['Outcome_dollar'].sum()
    return final_outcomes_df, outcome_counts, total_outcome, short_outcome, long_outcome, realized_outcome, unrealized_outcome

# --- Layout ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dcc.Store(id='prices-store'),
    dcc.Store(id='simulated-store'),
    dbc.Row([
        dbc.Col(html.H2("Trade Simulation, Price Download, Standardization, Outcome Analysis"), width=12)
    ], className="my-3"),
    dbc.Row([
        dbc.Col([
            dbc.Button("Download Prices from Yahoo Finance", id='download-prices-btn', color="secondary", className="mb-3"),
            html.Div(id='prices-table'),
            html.Div(id='prices-status'),
        ], width=12),
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Button("Run Simulation", id='simulate-btn', color="info", className="my-3"),
            html.Div(id='simulated-table'),
        ], width=12),
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Button("Standardize Trades", id='standardize-btn', color="warning", className="my-3"),
            html.Div(id='standardized-table'),
        ], width=12),
    ]),
    dbc.Row([
        dbc.Col([
            html.H5("Outcome Analysis"),
            html.Label("Select Ticker:"),
            dcc.Dropdown(
                id='ticker-dropdown',
                options=[{'label': 'All', 'value': 'ALL'}],  # will be updated dynamically
                value='ALL',
                clearable=False,
                style={'width': '200px'}
            ),
            dbc.Button("Analyze Outcomes", id='analyze-btn', color="success", className="my-3"),
            html.Div(id='outcome-analysis-table'),
            html.Div(id='outcome-summary'),
        ], width=12),
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='outcome-pie'), width=4),
        dbc.Col(dcc.Graph(id='sector-month-bar'), width=4),
        dbc.Col(dcc.Graph(id='realized-unrealized-bar'), width=4)
    ])
], fluid=True)

@app.callback(
    Output('prices-table', 'children'),
    Output('prices-status', 'children'),
    Output('prices-store', 'data'),
    Input('download-prices-btn', 'n_clicks'),
    prevent_initial_call=True
)
def download_prices_callback(n_clicks):
    prices_df = download_prices()
    if prices_df is None or prices_df.empty:
        return None, dbc.Alert("Failed to download prices!", color="danger"), None
    table = dash_table.DataTable(
        columns=[{"name": i, "id": i} for i in prices_df.columns],
        data=prices_df.head(10).to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'center'},
        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
        page_size=10
    )
    return table, dbc.Alert("Prices downloaded successfully!", color="success"), prices_df.to_json(date_format='iso', orient='split')

@app.callback(
    Output('simulated-table', 'children'),
    Output('simulated-store', 'data'),
    Output('ticker-dropdown', 'options'),
    Input('simulate-btn', 'n_clicks'),
    State('prices-store', 'data'),
    prevent_initial_call=True
)
def simulated_callback(n_clicks_simulate, prices_store_data):
    simulated_table = None
    simulated_store_data = None
    ticker_options = [{'label': 'All', 'value': 'ALL'}]
    if prices_store_data is None:
        simulated_table = dbc.Alert("Please download prices first!", color="danger")
    else:
        prices_df = pd.read_json(prices_store_data, orient='split')
        trades_df = simulate_trades(prices_df=prices_df)
        simulated_table = dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in trades_df.columns],
            data=trades_df.to_dict('records'),
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'center'},
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
            page_size=15
        )
        simulated_store_data = trades_df.to_json(date_format='iso', orient='split')
        ticker_options += [{'label': t, 'value': t} for t in sorted(trades_df['Ticker'].unique())]
    return simulated_table, simulated_store_data, ticker_options

@app.callback(
    Output('standardized-table', 'children'),
    Input('standardize-btn', 'n_clicks'),
    State('simulated-store', 'data'),
    prevent_initial_call=True
)
def standardize_callback(n_clicks, simulated_store_data):
    if simulated_store_data:
        executed_df = pd.read_json(simulated_store_data, orient='split')
        try:
            standardized_df = standardize_executed_trades(executed_df)
            return dash_table.DataTable(
                columns=[{"name": i, "id": i} for i in standardized_df.columns],
                data=standardized_df.to_dict('records'),
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'center'},
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                page_size=15
            )
        except Exception as e:
            return dbc.Alert(f"Standardization error: {e}", color="danger")
    else:
        return dbc.Alert("Please simulate executed trades first!", color="danger")

@app.callback(
    Output('outcome-analysis-table', 'children'),
    Output('outcome-summary', 'children'),
    Output('outcome-pie', 'figure'),
    Output('sector-month-bar', 'figure'),
    Output('realized-unrealized-bar', 'figure'),
    Input('analyze-btn', 'n_clicks'),
    State('simulated-store', 'data'),
    State('ticker-dropdown', 'value'),
    prevent_initial_call=True
)
def analyze_callback(n_clicks, simulated_store_data, ticker):
    if simulated_store_data:
        executed_df = pd.read_json(simulated_store_data, orient='split')
        try:
            standardized_df = standardize_executed_trades(executed_df)
            # Assign sector from trading-data.csv or dummy if missing
            trading_data = pd.read_csv('trading-data.csv')
            if 'sector' in trading_data.columns:
                sector_map = dict(zip(trading_data['ticker'], trading_data['sector']))
                standardized_df['Sector'] = standardized_df['Ticker'].map(sector_map)
            else:
                sectors = ['Tech', 'Finance', 'Healthcare', 'Energy']
                tickers = standardized_df['Ticker'].unique()
                sector_map = {t: sectors[i % len(sectors)] for i, t in enumerate(tickers)}
                standardized_df['Sector'] = standardized_df['Ticker'].map(sector_map)

            # Analyze outcomes for all months
            months = ['April', 'May', 'June']
            all_final_outcomes = []
            all_outcome_counts = {}
            total_outcome = 0
            total_short = 0
            total_long = 0
            total_realized = 0
            total_unrealized = 0
            for month in months:
                final_outcomes_df, outcome_counts, month_total, short_outcome, long_outcome, realized_outcome, unrealized_outcome = analyze_outcomes(standardized_df, month, ticker)
                final_outcomes_df['Month'] = month
                all_final_outcomes.append(final_outcomes_df)
                all_outcome_counts[month] = outcome_counts
                total_outcome += month_total
                total_short += short_outcome
                total_long += long_outcome
                total_realized += realized_outcome
                total_unrealized += unrealized_outcome
            all_final_outcomes_df = pd.concat(all_final_outcomes, ignore_index=True)
            if all_final_outcomes_df.empty:
                return dbc.Alert("No initial positions found for these months/ticker.", color="warning"), None, {}, {}, {}
            outcome_analysis_table = dash_table.DataTable(
                columns=[{"name": i, "id": i} for i in all_final_outcomes_df.columns],
                data=all_final_outcomes_df.to_dict('records'),
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'center'},
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                page_size=15
            )
            # Summary
            summary_items = []
            for month in months:
                summary_items.append(html.Li(
                    f"{month} - Failed: {all_outcome_counts[month].get('failed', 0)}, "
                    f"Succeeded: {all_outcome_counts[month].get('succeeded', 0)}, "
                    f"Unknown: {all_outcome_counts[month].get('unknown', 0)}"
                ))
            summary_items.append(html.Li(f"Total outcome ($): {total_outcome}"))
            summary_items.append(html.Li(f"Short outcomes ($): {total_short}"))
            summary_items.append(html.Li(f"Long outcomes ($): {total_long}"))
            summary_items.append(html.Li(f"Total realized P&L ($): {total_realized}"))
            summary_items.append(html.Li(f"Total unrealized P&L ($): {total_unrealized}"))
            summary_items.append(html.Li(f"Total net P&L ($): {total_realized + total_unrealized}"))
            outcome_summary = html.Div([
                html.H5("Outcome Summary"),
                html.Ul(summary_items)
            ])
            # Pie chart: all months together
            pie_fig = px.pie(
                all_final_outcomes_df,
                names='Outcome',
                title=f'April, May & June Initial Position Outcomes',
                color='Outcome',
                color_discrete_map={'failed': '#e74c3c', 'succeeded': '#2ecc71', 'unknown': '#95a5a6'},
                hole=0.4
            )
            pie_fig.update_traces(textinfo='percent+label', pull=[0.05, 0.05, 0.05])
            pie_fig.update_layout(paper_bgcolor='#fff', plot_bgcolor='#fff')
            # --- Monthly sector P&L bar chart ---
            sector_month_df = []
            for month in months:
                month_df = standardized_df[standardized_df['Month'] == month]
                sector_sum = month_df.groupby('Sector')['Standardized_Trade'].sum().reset_index()
                sector_sum['Month'] = month
                sector_month_df.append(sector_sum)
            sector_month_df = pd.concat(sector_month_df, ignore_index=True)
            sector_month_bar = px.bar(
                sector_month_df,
                x='Sector',
                y='Standardized_Trade',
                color='Month',
                barmode='group',
                title='Monthly Sector P&L (April, May, June)'
            )
            sector_month_bar.update_layout(
                paper_bgcolor='#fff',
                plot_bgcolor='#fff',
                legend_title_text='Month',
                xaxis_title='Sector',
                yaxis_title='P&L ($)'
            )
            # Realized vs Unrealized by ticker
            realized_unrealized_fig = px.bar(
                all_final_outcomes_df.groupby(['Ticker', 'Realized'])['Outcome_dollar'].sum().reset_index(),
                x='Ticker',
                y='Outcome_dollar',
                color='Realized',
                barmode='group',
                title='Realized vs Unrealized P&L by Ticker',
                color_discrete_map={True: '#2ecc71', False: '#e67e22'}
            )
            realized_unrealized_fig.update_layout(paper_bgcolor='#fff', plot_bgcolor='#fff')
            return outcome_analysis_table, outcome_summary, pie_fig, sector_month_bar, realized_unrealized_fig
        except Exception as e:
            return dbc.Alert(f"Analysis error: {e}", color="danger"), None, {}, {}, {}
    else:
        return dbc.Alert("Please simulate executed trades first!", color="danger"), None, {}, {}, {}

if __name__ == '__main__':
    app.run(debug=True)