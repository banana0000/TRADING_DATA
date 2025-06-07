import dash
from dash import dcc, html, Input, Output, State, callback, dash_table
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

app = dash.Dash(__name__)

# Popular stock symbols for dropdown
POPULAR_STOCKS = [
    {'label': 'Apple Inc. (AAPL)', 'value': 'AAPL'},
    {'label': 'Microsoft Corp (MSFT)', 'value': 'MSFT'},
    {'label': 'Amazon.com Inc (AMZN)', 'value': 'AMZN'},
    {'label': 'Alphabet Inc (GOOGL)', 'value': 'GOOGL'},
    {'label': 'Tesla Inc (TSLA)', 'value': 'TSLA'},
    {'label': 'Meta Platforms Inc (META)', 'value': 'META'},
    {'label': 'NVIDIA Corp (NVDA)', 'value': 'NVDA'},
    {'label': 'Netflix Inc (NFLX)', 'value': 'NFLX'},
    {'label': 'Spotify Technology (SPOT)', 'value': 'SPOT'},
    {'label': 'Adobe Inc (ADBE)', 'value': 'ADBE'},
    {'label': 'PayPal Holdings (PYPL)', 'value': 'PYPL'},
    {'label': 'Intel Corp (INTC)', 'value': 'INTC'},
    {'label': 'Cisco Systems (CSCO)', 'value': 'CSCO'},
    {'label': 'Oracle Corp (ORCL)', 'value': 'ORCL'},
    {'label': 'Salesforce Inc (CRM)', 'value': 'CRM'}
]

# Time periods for dropdown
TIME_PERIODS = [
    {'label': '1 Day', 'value': '1d'},
    {'label': '5 Days', 'value': '5d'},
    {'label': '1 Month', 'value': '1mo'},
    {'label': '3 Months', 'value': '3mo'},
    {'label': '6 Months', 'value': '6mo'},
    {'label': '1 Year', 'value': '1y'},
    {'label': '2 Years', 'value': '2y'},
    {'label': '5 Years', 'value': '5y'},
    {'label': '10 Years', 'value': '10y'},
    {'label': 'Year to Date', 'value': 'ytd'},
    {'label': 'Max', 'value': 'max'}
]

app.layout = html.Div([
    html.H1("Multi-Stock Yahoo Finance Dashboard", 
            style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '30px'}),
    
    html.Div([
        html.Div([
            html.Label("Select Stocks (Multiple):", style={'fontWeight': 'bold', 'marginBottom': '10px'}),
            dcc.Dropdown(
                id='ticker-dropdown',
                options=POPULAR_STOCKS,
                value=['AAPL', 'MSFT'],  # Default multiple selection
                multi=True,  # Enable multiple selection
                placeholder="Select stocks...",
                style={'marginBottom': '10px'}
            ),
            html.Div([
                dcc.Input(
                    id='custom-ticker-input',
                    type='text',
                    placeholder='Add custom ticker (e.g., GOOG)',
                    style={'width': '70%', 'marginRight': '10px', 'padding': '8px'}
                ),
                html.Button('Add Ticker', id='add-ticker-btn', n_clicks=0,
                           style={'width': '25%', 'padding': '8px', 'backgroundColor': '#27ae60', 
                                 'color': 'white', 'border': 'none', 'borderRadius': '3px'})
            ], style={'marginTop': '10px'})
        ], style={'width': '45%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        
        html.Div([
            html.Label("Time Periods (Multiple):", style={'fontWeight': 'bold', 'marginBottom': '10px'}),
            dcc.Dropdown(
                id='period-dropdown',
                options=TIME_PERIODS,
                value=['1mo', '3mo'],  # Default multiple selection
                multi=True,  # Enable multiple selection
                placeholder="Select time periods...",
                style={'marginBottom': '10px'}
            )
        ], style={'width': '45%', 'display': 'inline-block', 'marginLeft': '10%'}),
        
        html.Button('Update Dashboard', id='update-button', n_clicks=0,
                   style={'width': '100%', 'padding': '10px', 'backgroundColor': '#3498db', 
                         'color': 'white', 'border': 'none', 'borderRadius': '5px',
                         'fontSize': '16px', 'marginTop': '20px'})
    ], style={'margin': '20px', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '10px'}),
    
    # Summary section
    html.Div(id='summary-section', style={'margin': '20px'}),
    
    dcc.Tabs([
        dcc.Tab(label='📈 Price Comparison', children=[
            html.Div(id='comparison-charts')
        ]),
        
        dcc.Tab(label='📊 Individual Analysis', children=[
            html.Div(id='individual-analysis')
        ]),
        
        dcc.Tab(label='📰 News', children=[
            html.Div(id='news-section')
        ]),
        
        dcc.Tab(label='📋 Financial Comparison', children=[
            html.Div(id='financial-comparison')
        ])
    ])
])

# Callback to add custom ticker
@callback(
    Output('ticker-dropdown', 'options'),
    Output('ticker-dropdown', 'value'),
    Input('add-ticker-btn', 'n_clicks'),
    State('custom-ticker-input', 'value'),
    State('ticker-dropdown', 'options'),
    State('ticker-dropdown', 'value')
)
def add_custom_ticker(n_clicks, custom_ticker, current_options, current_values):
    if n_clicks > 0 and custom_ticker:
        ticker = custom_ticker.upper().strip()
        if ticker and ticker not in [opt['value'] for opt in current_options]:
            # Verify ticker exists
            try:
                test_stock = yf.Ticker(ticker)
                info = test_stock.info
                if info.get('longName') or info.get('shortName'):
                    new_option = {'label': f'{info.get("longName", ticker)} ({ticker})', 'value': ticker}
                    updated_options = current_options + [new_option]
                    updated_values = (current_values or []) + [ticker]
                    return updated_options, updated_values
            except:
                pass
    return current_options, current_values

# Main dashboard update callback
@callback(
    [Output('summary-section', 'children'),
     Output('comparison-charts', 'children'),
     Output('individual-analysis', 'children'),
     Output('news-section', 'children'),
     Output('financial-comparison', 'children')],
    [Input('update-button', 'n_clicks')],
    [State('ticker-dropdown', 'value'),
     State('period-dropdown', 'value')]
)
def update_dashboard(n_clicks, tickers, periods):
    if not tickers or not periods:
        error_msg = "Please select at least one stock and one time period"
        return error_msg, error_msg, error_msg, error_msg, error_msg
    
    # Ensure tickers and periods are lists
    if isinstance(tickers, str):
        tickers = [tickers]
    if isinstance(periods, str):
        periods = [periods]
    
    try:
        # Summary Section
        summary_cards = []
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                current_price = info.get('currentPrice', 'N/A')
                change = info.get('regularMarketChange', 'N/A')
                change_percent = info.get('regularMarketChangePercent', 'N/A')
                
                summary_cards.append(
                    html.Div([
                        html.H4(f"{ticker}", style={'margin': '0', 'color': '#2c3e50'}),
                        html.H3(f"${current_price}", style={'margin': '5px 0', 'color': '#3498db'}),
                        html.P(f"Change: {change} ({change_percent:.2f}%)" if isinstance(change_percent, (int, float)) else f"Change: {change}",
                               style={'margin': '0', 'color': '#e74c3c' if isinstance(change, (int, float)) and change < 0 else '#27ae60'})
                    ], style={
                        'backgroundColor': '#ffffff',
                        'padding': '15px',
                        'borderRadius': '8px',
                        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                        'margin': '10px',
                        'width': '200px',
                        'display': 'inline-block',
                        'textAlign': 'center'
                    })
                )
            except Exception as e:
                summary_cards.append(
                    html.Div([
                        html.H4(f"{ticker}", style={'margin': '0', 'color': '#e74c3c'}),
                        html.P("Error loading data", style={'margin': '0', 'color': '#7f8c8d'})
                    ], style={
                        'backgroundColor': '#ffffff',
                        'padding': '15px',
                        'borderRadius': '8px',
                        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                        'margin': '10px',
                        'width': '200px',
                        'display': 'inline-block',
                        'textAlign': 'center'
                    })
                )
        
        summary_section = html.Div([
            html.H3("Quick Summary", style={'color': '#2c3e50', 'marginBottom': '20px'}),
            html.Div(summary_cards)
        ])
        
        # Comparison Charts
        comparison_charts = []
        colors = ['#3498db', '#e74c3c', '#27ae60', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#e67e22']
        
        for period in periods:
            comparison_fig = go.Figure()
            
            for i, ticker in enumerate(tickers):
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period=period)
                    
                    if not hist.empty:
                        # Normalize prices to percentage change for comparison
                        normalized_prices = ((hist['Close'] / hist['Close'].iloc[0]) - 1) * 100
                        
                        comparison_fig.add_trace(go.Scatter(
                            x=hist.index,
                            y=normalized_prices,
                            mode='lines',
                            name=ticker,
                            line=dict(color=colors[i % len(colors)], width=2)
                        ))
                except Exception as e:
                    continue
            
            comparison_fig.update_layout(
                title=f'Stock Performance Comparison - {period.upper()}',
                xaxis_title="Date",
                yaxis_title="Percentage Change (%)",
                template='plotly_white',
                showlegend=True,
                height=400
            )
            
            comparison_charts.append(
                html.Div([
                    dcc.Graph(figure=comparison_fig)
                ], style={'margin': '20px 0'})
            )
        
        comparison_section = html.Div(comparison_charts)
        
        # Individual Analysis
        individual_analysis = []
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                # Use the first period for individual analysis
                hist = stock.history(period=periods[0])
                
                if not hist.empty:
                    # Price chart with volume
                    price_fig = go.Figure()
                    price_fig.add_trace(go.Scatter(
                        x=hist.index,
                        y=hist['Close'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color='#3498db', width=2)
                    ))
                    price_fig.update_layout(
                        title=f'{ticker} - Price Chart ({periods[0].upper()})',
                        xaxis_title="Date",
                        yaxis_title="Price (USD)",
                        template='plotly_white',
                        height=300
                    )
                    
                    # Volume chart
                    volume_fig = go.Figure()
                    volume_fig.add_trace(go.Bar(
                        x=hist.index,
                        y=hist['Volume'],
                        name='Volume',
                        marker_color='rgba(255, 99, 132, 0.7)'
                    ))
                    volume_fig.update_layout(
                        title=f'{ticker} - Volume',
                        xaxis_title="Date",
                        yaxis_title="Volume",
                        template='plotly_white',
                        height=300
                    )
                    
                    individual_analysis.append(
                        html.Div([
                            html.H3(f"{ticker} Analysis", style={'color': '#2c3e50', 'marginTop': '30px'}),
                            html.Div([
                                html.Div([
                                    dcc.Graph(figure=price_fig)
                                ], style={'width': '50%', 'display': 'inline-block'}),
                                html.Div([
                                    dcc.Graph(figure=volume_fig)
                                ], style={'width': '50%', 'display': 'inline-block'})
                            ])
                        ])
                    )
            except Exception as e:
                individual_analysis.append(
                    html.Div([
                        html.H3(f"{ticker} Analysis", style={'color': '#e74c3c'}),
                        html.P(f"Error loading data: {str(e)}")
                    ])
                )
        
        individual_section = html.Div(individual_analysis)
        
        # News Section
        news_section = []
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                news = stock.news[:3]  # Limit to 3 news items per stock
                
                ticker_news = [html.H3(f"{ticker} News", style={'color': '#2c3e50', 'marginTop': '20px'})]
                
                for item in news:
                    ticker_news.append(
                        html.Div([
                            html.H4(
                                html.A(item['title'], href=item['link'], target='_blank',
                                      style={'color': '#2c3e50', 'textDecoration': 'none'}),
                                style={'marginBottom': '10px'}
                            ),
                            html.P(item.get('summary', 'No summary available')[:200] + "...",
                                  style={'color': '#7f8c8d', 'lineHeight': '1.6'}),
                            html.Small(f"Source: {item.get('publisher', 'Unknown')}",
                                      style={'color': '#95a5a6'})
                        ], style={
                            'backgroundColor': '#ffffff',
                            'margin': '10px 0',
                            'padding': '15px',
                            'borderRadius': '8px',
                            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                            'borderLeft': '4px solid #3498db'
                        })
                    )
                
                news_section.extend(ticker_news)
            except Exception as e:
                news_section.append(
                    html.Div([
                        html.H3(f"{ticker} News", style={'color': '#e74c3c'}),
                        html.P(f"Unable to load news: {str(e)}")
                    ])
                )
        
        news_div = html.Div(news_section)
        
        # Financial Comparison
        financial_data = []
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                financial_data.append({
                    'Ticker': ticker,
                    'Market Cap': f"${info.get('marketCap', 0):,}",
                    'P/E Ratio': info.get('trailingPE', 'N/A'),
                    'Revenue': f"${info.get('totalRevenue', 0):,}",
                    'Profit Margin': f"{info.get('profitMargins', 0)*100:.2f}%" if info.get('profitMargins') else 'N/A',
                    'Beta': info.get('beta', 'N/A'),
                    'Dividend Yield': f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else 'N/A'
                })
            except Exception as e:
                financial_data.append({
                    'Ticker': ticker,
                    'Market Cap': 'Error',
                    'P/E Ratio': 'Error',
                    'Revenue': 'Error',
                    'Profit Margin': 'Error',
                    'Beta': 'Error',
                    'Dividend Yield': 'Error'
                })
        
        financial_table = dash_table.DataTable(
            data=financial_data,
            columns=[{"name": i, "id": i} for i in financial_data[0].keys()],
            style_cell={'textAlign': 'left', 'padding': '10px'},
            style_header={'backgroundColor': '#3498db', 'color': 'white', 'fontWeight': 'bold'},
            style_data={'backgroundColor': '#f8f9fa'},
            style_table={'margin': '20px'},
            sort_action='native'
        )
        
        return summary_section, comparison_section, individual_section, news_div, financial_table
        
    except Exception as e:
        error_msg = f"Error updating dashboard: {str(e)}"
        return error_msg, error_msg, error_msg, error_msg, error_msg

if __name__ == '__main__':
    app.run(debug=True)