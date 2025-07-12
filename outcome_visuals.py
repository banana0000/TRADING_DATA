import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

# --- 1. Load and preprocess the data ---

df = pd.read_csv("standardized-executed-trades.csv")
df['Date'] = pd.to_datetime(df['Date'])
if 'Month' not in df.columns:
    df['Month'] = df['Date'].dt.strftime('%Y-%m')
df_full = df.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)

# --- 2. Function to generate final_outcomes_df for a given month and ticker ---

def analyze_month(selected_month, selected_ticker):
    month_trades = df[df['Month'] == selected_month].copy()
    if selected_ticker and selected_ticker != "All":
        month_trades = month_trades[month_trades['Ticker'] == selected_ticker]
    target_actions = ['Initial Buy', 'Initial Short']
    filtered_actions = month_trades[month_trades['Action'].isin(target_actions)]
    action_counts = filtered_actions.groupby(['Ticker', 'Action']).size()
    result_df = action_counts.unstack(fill_value=0)
    if 'Initial Buy' not in result_df.columns:
        result_df['Initial Buy'] = 0
    if 'Initial Short' not in result_df.columns:
        result_df['Initial Short'] = 0
    result_df = result_df[['Initial Buy', 'Initial Short']]
    tickers_from_original_result_df = result_df.reset_index()['Ticker'].unique()
    initial_positions_to_analyze = df_full[
        (df_full['Month'] == selected_month) &
        (df_full['Action'].isin(['Initial Buy', 'Initial Short'])) &
        (df_full['Ticker'].isin(tickers_from_original_result_df))
    ].copy()
    if selected_ticker and selected_ticker != "All":
        initial_positions_to_analyze = initial_positions_to_analyze[
            initial_positions_to_analyze['Ticker'] == selected_ticker
        ]
    outcome_details_list = []
    for _, initial_trade_row in initial_positions_to_analyze.iterrows():
        current_ticker = initial_trade_row['Ticker']
        initial_action_type = initial_trade_row['Action']
        initial_trade_original_index = initial_trade_row.name
        subsequent_trades_for_ticker_all = df_full[
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
        if initial_action_type == 'Initial Buy':
            if len(actions_list) >= 1:
                action1 = actions_list[0]
                if action1 == 'Stop-Loss Sell':
                    outcome = 'failed'
                    outcome_dollar = -27
                elif action1 == 'PT1 Sell':
                    if len(actions_list) == 2:
                        action2 = actions_list[1]
                        if action2 == 'Stop-Loss Sell':
                            outcome = 'succeeded'
                            outcome_dollar = 13
                        elif action2 == 'PT2 Sell':
                            outcome = 'succeeded'
                            outcome_dollar = 36
                    elif len(actions_list) == 3:
                        action3 = actions_list[2]
                        if action3 == 'Stop-Loss Sell':
                            outcome = 'succeeded'
                            outcome_dollar = 49
                        elif action3 == 'PT3 Sell':
                            outcome = 'succeeded'
                            outcome_dollar = 74
                    else:
                        outcome = 'succeeded'
                        outcome_dollar = 13
        elif initial_action_type == 'Initial Short':
            if len(actions_list) >= 1:
                action1 = actions_list[0]
                if action1 == 'Stop-Loss Buy':
                    outcome = 'failed'
                    outcome_dollar = -27
                elif action1 == 'PT1 Buy':
                    if len(actions_list) == 2:
                        action2 = actions_list[1]
                        if action2 == 'Stop-Loss Buy':
                            outcome = 'succeeded'
                            outcome_dollar = 13
                        elif action2 == 'PT2 Buy':
                            outcome = 'succeeded'
                            outcome_dollar = 36
                    elif len(actions_list) == 3:
                        action3 = actions_list[2]
                        if action3 == 'Stop-Loss Buy':
                            outcome = 'succeeded'
                            outcome_dollar = 49
                        elif action3 == 'PT3 Buy':
                            outcome = 'succeeded'
                            outcome_dollar = 74
                    else:
                        outcome = 'succeeded'
                        outcome_dollar = 13
        outcome_details_list.append({
            'Date_of_Initial_Action': initial_trade_row['Date'],
            'Ticker': initial_trade_row['Ticker'],
            'Initial_Action_Type': initial_action_type,
            'Initial_Price': initial_trade_row['Price'],
            'Outcome': outcome,
            'Outcome_dollar': outcome_dollar
        })
    final_outcomes_df = pd.DataFrame(outcome_details_list)
    return final_outcomes_df

# --- 3. Dash app setup ---

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

available_months = sorted(df['Month'].unique())
default_month = available_months[0] if len(available_months) > 0 else None
available_tickers = sorted(df['Ticker'].unique())
default_ticker = "All"

outcome_colors = {
    'succeeded': '#2ecc71',  # green
    'failed': '#e74c3c',     # red
    'unknown': '#95a5a6'     # gray
}
type_colors = {
    'Initial Buy': '#3498db',    # blue
    'Initial Short': '#f39c12'   # orange
}

def dark_fig(fig):
    fig.update_layout(
        paper_bgcolor='#111',
        plot_bgcolor='#111',
        font=dict(family='Arial', size=15, color='white'),
        legend=dict(font=dict(color='white')),
        title_font_color='white',
        xaxis=dict(showgrid=True, gridcolor='#444'),
        yaxis=dict(showgrid=True, gridcolor='#444')
    )
    return fig

def kpi_card(title, value, color):
    return dbc.Card(
        dbc.CardBody([
            html.H6(title, className="card-title text-center", style={"color": "#fff"}),
            html.H3(value, className="card-text text-center", style={"color": color, "fontWeight": "bold"})
        ]),
        className="m-1",
        style={"backgroundColor": "#222", "borderRadius": "12px"}
    )

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(
            html.Div(
                html.H1("Position Outcome Analysis by Month",
                        style={"color": "white", "background": "#111", "padding": "18px 0", "marginBottom": 0, "textAlign": "center", "borderRadius": "12px"}),
                style={"background": "#111", "borderRadius": "12px", "marginBottom": "28px"}
            ),
            width=12
        )
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Label("Select Month:", style={"color": "white", "fontWeight": "bold", "fontSize": "1.1rem"}),
            dbc.Select(
                id='month-dropdown',
                options=[{'label': m, 'value': m} for m in available_months],
                value=default_month,
                className="bg-dark text-light border-secondary"
            ),
        ], width=2),
        dbc.Col([
            dbc.Label("Select Stock:", style={"color": "white", "fontWeight": "bold", "fontSize": "1.1rem"}),
            dbc.Select(
                id='ticker-dropdown',
                options=[{'label': 'All', 'value': 'All'}] +
                        [{'label': t, 'value': t} for t in available_tickers],
                value=default_ticker,
                className="bg-dark text-light border-secondary"
            ),
        ], width=2),
        dbc.Col([
            dbc.Label("Filter by Outcome:", style={"color": "white", "fontWeight": "bold", "fontSize": "1.1rem"}),
            dcc.Checklist(
                id='outcome-filter',
                options=[
                    {'label': 'Succeeded', 'value': 'succeeded'},
                    {'label': 'Failed', 'value': 'failed'},
                    {'label': 'Unknown', 'value': 'unknown'}
                ],
                value=['succeeded', 'failed', 'unknown'],
                inline=True,
                inputStyle={"margin-right": "5px", "margin-left": "10px"},
                labelStyle={"margin-right": "15px", "color": "white"}
            )
        ], width=4),
        dbc.Col([
            dbc.Label("P&L Range ($):", style={"color": "white", "fontWeight": "bold", "fontSize": "1.1rem"}),
            dcc.RangeSlider(
                id='pnl-slider',
                min=-100, max=100, step=1, value=[-100, 100],
                marks={-100: '-100', 0: '0', 100: '100'},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], width=4)
    ], className="mb-4 justify-content-center"),
    dbc.Row([
        dbc.Col(
            dbc.Row(id="kpi-row", className="mb-3 g-2 justify-content-center"),
            width=12
        )
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='pie-chart'), width=6),
        dbc.Col(dcc.Graph(id='bar-chart'), width=6)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='long-short-chart'), width=12)
    ], className="mt-4"),
    dbc.Row([
        dbc.Col(dcc.Graph(id='cumulative-pnl-chart'), width=12)
    ], className="mt-4"),
    html.Br(),
    dbc.Row([
        dbc.Col(
            html.Div(
                html.H2("Detailed Position Outcomes", className="text-center", style={"color": "white", "background": "#111", "padding": "12px 0", "borderRadius": "12px"}),
                style={"background": "#111", "borderRadius": "12px"}
            ),
            width=12
        )
    ]),
    html.Br(),
    html.Br(),
    dbc.Row([
        dbc.Col(
            dash_table.DataTable(
                id='details-table',
                columns=[
                    {"name": "Date", "id": "Date_of_Initial_Action"},
                    {"name": "Ticker", "id": "Ticker"},
                    {"name": "Type", "id": "Initial_Action_Type"},
                    {"name": "Price", "id": "Initial_Price"},
                    {"name": "Outcome", "id": "Outcome"},
                    {"name": "P&L ($)", "id": "Outcome_dollar"},
                ],
                page_size=15,
                style_table={'overflowX': 'auto', 'margin': 'auto', 'maxWidth': '900px'},
                style_cell={
                    'textAlign': 'left',
                    'padding': '6px',
                    'fontFamily': 'Arial',
                    'backgroundColor': '#111',
                    'color': 'white'
                },
                style_header={
                    'backgroundColor': '#222',
                    'fontWeight': 'bold',
                    'fontSize': 16,
                    'color': 'white'
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': '#222'
                    },
                    {
                        'if': {'column_id': 'Outcome', 'filter_query': '{Outcome} = "succeeded"'},
                        'color': outcome_colors['succeeded'],
                        'fontWeight': 'bold'
                    },
                    {
                        'if': {'column_id': 'Outcome', 'filter_query': '{Outcome} = "failed"'},
                        'color': outcome_colors['failed'],
                        'fontWeight': 'bold'
                    },
                    {
                        'if': {'column_id': 'Outcome', 'filter_query': '{Outcome} = "unknown"'},
                        'color': outcome_colors['unknown'],
                        'fontWeight': 'bold'
                    },
                    {
                        'if': {'column_id': 'Outcome_dollar', 'filter_query': '{Outcome_dollar} < 0'},
                        'color': '#e74c3c',
                        'fontWeight': 'bold'
                    },
                    {
                        'if': {'column_id': 'Outcome_dollar', 'filter_query': '{Outcome_dollar} >= 0'},
                        'color': '#2ecc71',
                        'fontWeight': 'bold'
                    },
                ],
            ),
            width=12
        )
    ])
], fluid=True, style={'backgroundColor': '#111', 'minHeight': '100vh', 'paddingBottom': 40})

# --- 4. Callbacks for interactivity ---

@app.callback(
    [
        Output('pie-chart', 'figure'),
        Output('bar-chart', 'figure'),
        Output('long-short-chart', 'figure'),
        Output('details-table', 'data'),
        Output('kpi-row', 'children'),
        Output('cumulative-pnl-chart', 'figure'),
    ],
    [
        Input('month-dropdown', 'value'),
        Input('ticker-dropdown', 'value'),
        Input('outcome-filter', 'value'),
        Input('pnl-slider', 'value')
    ]
)
def update_dashboard(selected_month, selected_ticker, selected_outcomes, pnl_range):
    empty_fig = dark_fig(px.scatter())
    empty_kpis = [
        dbc.Col(kpi_card("Total P&L", "$0", "#cccccc"), width=2),
        dbc.Col(kpi_card("Succeeded", 0, "#cccccc"), width=2),
        dbc.Col(kpi_card("Failed", 0, "#cccccc"), width=2),
        dbc.Col(kpi_card("Unknown", 0, "#cccccc"), width=2),
        dbc.Col(kpi_card("Winrate", "N/A", "#cccccc"), width=2),
    ]
    if not selected_month:
        return empty_fig, empty_fig, empty_fig, [], empty_kpis, empty_fig

    final_outcomes_df = analyze_month(selected_month, selected_ticker)

    filtered_df = final_outcomes_df[
        final_outcomes_df['Outcome'].isin(selected_outcomes) &
        (final_outcomes_df['Outcome_dollar'] >= pnl_range[0]) &
        (final_outcomes_df['Outcome_dollar'] <= pnl_range[1])
    ]

    if filtered_df.empty:
        return empty_fig, empty_fig, empty_fig, [], empty_kpis, empty_fig

    # Pie chart
    fig_pie = px.pie(
        filtered_df,
        names='Outcome',
        title=f'{selected_month} Initial Positions Outcomes',
        color='Outcome',
        color_discrete_map=outcome_colors,
        hole=0.4
    )
    fig_pie.update_traces(textinfo='percent+label', pull=[0.05, 0.05, 0.05])
    fig_pie = dark_fig(fig_pie)

    # Bar chart: Outcome P&L
    fig_bar = px.bar(
        filtered_df.groupby('Outcome')['Outcome_dollar'].sum().reset_index(),
        x='Outcome',
        y='Outcome_dollar',
        color='Outcome',
        title=f'Total P&L by Outcome ({selected_month})',
        color_discrete_map=outcome_colors
    )
    fig_bar = dark_fig(fig_bar)

    # Bar chart: Long vs Short
    fig_long_short = px.bar(
        filtered_df.groupby('Initial_Action_Type')['Outcome_dollar'].sum().reset_index(),
        x='Initial_Action_Type',
        y='Outcome_dollar',
        color='Initial_Action_Type',
        title=f'Total P&L: Long vs Short ({selected_month})',
        color_discrete_map=type_colors
    )
    fig_long_short = dark_fig(fig_long_short)

    # Cumulative P&L chart
    cum_df = filtered_df.sort_values('Date_of_Initial_Action')
    cum_df['Cumulative_PnL'] = cum_df['Outcome_dollar'].cumsum()
    fig_cum = px.line(
        cum_df,
        x='Date_of_Initial_Action',
        y='Cumulative_PnL',
        title='Cumulative P&L Over Time'
    )
    fig_cum = dark_fig(fig_cum)

    # Table data
    table_data = filtered_df.to_dict('records')

    # KPIs
    total_pnl = filtered_df['Outcome_dollar'].sum()
    succeeded = (filtered_df['Outcome'] == 'succeeded').sum()
    failed = (filtered_df['Outcome'] == 'failed').sum()
    unknown = (filtered_df['Outcome'] == 'unknown').sum()
    total = len(filtered_df)
    winrate = f"{(succeeded / total * 100):.1f}%" if total > 0 else "N/A"
    kpis = [
        dbc.Col(kpi_card("Total P&L", f"${total_pnl}", "#2ecc71" if total_pnl >= 0 else "#e74c3c"), width=2),
        dbc.Col(kpi_card("Succeeded", succeeded, "#2ecc71"), width=2),
        dbc.Col(kpi_card("Failed", failed, "#e74c3c"), width=2),
        dbc.Col(kpi_card("Unknown", unknown, "#95a5a6"), width=2),
        dbc.Col(kpi_card("Winrate", winrate, "#2ecc71" if succeeded >= failed else "#e74c3c"), width=2),
    ]

    return (
        fig_pie, fig_bar, fig_long_short, table_data, kpis, fig_cum
    )

if __name__ == '__main__':
    app.run(port=8000)