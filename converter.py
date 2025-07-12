import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
import requests
import time
from dotenv import load_dotenv
import os

# --- .env betöltése ---
load_dotenv()

# --- Színek és stílusok ---
COLOR_DARKBLUE = "#273F4F"
COLOR_ORANGE = "#FE7743"
COLOR_GRAY = "#D7D7D7"
COLOR_BLACK = "#222"

button_orange = {
    "background": COLOR_ORANGE,
    "color": "#fff",
    "fontWeight": "bold",
    "fontSize": "1.3rem",
    "border": f"2px solid {COLOR_DARKBLUE}",
    "boxShadow": f"0 2px 8px {COLOR_ORANGE}88",
    "borderRadius": "10px",
    "height": "60px"
}
dropdown_style = {
    "background": COLOR_GRAY,
    "color": COLOR_BLACK,
    "border": f"1.5px solid {COLOR_DARKBLUE}",
    "fontWeight": "bold",
    "fontSize": "1.1rem",
    "borderRadius": "8px"
}
input_style = {
    "background": "rgba(255,255,255,0.8)",
    "fontSize": "1.2rem",
    "fontWeight": "bold",
    "borderRadius": "8px"
}

# --- Currency API config ---
EXCHANGE_API_KEY = os.getenv("EXCHANGE_API_KEY")  # <-- .env-ből olvassuk be
if not EXCHANGE_API_KEY:
    raise ValueError("Hiányzik az EXCHANGE_API_KEY környezeti változó!")

EXCHANGE_BASE_URL = f"https://v6.exchangerate-api.com/v6/{EXCHANGE_API_KEY}/latest/"

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(
    [
        html.H2("Currency Calculator", style={
            "marginTop": "40px",
            "textAlign": "center",
            "fontWeight": "bold",
            "color": COLOR_DARKBLUE,
            "letterSpacing": "2px"
        }),
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardBody([
                            html.Div([
                                dbc.Input(
                                    id="cc-amount",
                                    type="number",
                                    min=0,
                                    step=0.01,
                                    value=1,
                                    placeholder="Amount",
                                    size="lg",
                                    style=input_style
                                ),
                            ], style={"marginBottom": "18px"}),
                            html.Div([
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
                            ], style={"marginBottom": "18px"}),
                            html.Div([
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
                            ], style={"marginBottom": "28px"}),
                            dbc.Button(
                                "Convert",
                                id="cc-convert-btn",
                                n_clicks=0,
                                className="w-100",
                                style=button_orange
                            ),
                        ])
                    ],
                    style={
                        "maxWidth": "350px",
                        "margin": "auto",
                        "marginTop": "40px",
                        "padding": "30px 20px",
                        "boxShadow": "0 4px 24px #8ec5fc55, 0 1.5px 8px #e0c3fc55",
                        "borderRadius": "18px",
                        "background": "linear-gradient(120deg, #f8fafc 80%, #e0c3fc 100%)",
                        "border": f"2px solid {COLOR_DARKBLUE}"
                    }
                ),
                width=12
            )
        ], justify="center"),
        # Modal
        dbc.Modal(
            [
                dbc.ModalHeader(
                    dbc.ModalTitle("Currency Conversion Result"),
                    style={"background": COLOR_DARKBLUE, "color": "#fff"}
                ),
                dbc.ModalBody(
                    dbc.Spinner(
                        html.Div(id="cc-modal-body", style={
                            "fontWeight": "bold",
                            "fontSize": "2rem",
                            "textAlign": "center",
                            "color": COLOR_DARKBLUE,
                            "padding": "20px"
                        })
                    )
                ),
                dbc.ModalFooter(
                    dbc.Button("Close", id="cc-modal-close", className="ms-auto", n_clicks=0, style=button_orange)
                ),
            ],
            id="cc-modal",
            is_open=False,
            centered=True,
            size="md",
            style={"borderRadius": "18px"}
        ),
    ],
    fluid=True
)

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
            return True, "Please enter a valid amount and select different currencies."
        time.sleep(1)
        url = f"{EXCHANGE_BASE_URL}{from_curr}"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                return True, "API error!"
            data = response.json()
            rate = data["conversion_rates"].get(to_curr)
            if not rate:
                return True, "Currency not supported!"
            converted = amount * rate
            return True, f"{amount} {from_curr} = {converted:.2f} {to_curr}"
        except Exception:
            return True, "Network or API error!"
    elif trigger_id == "cc-modal-close":
        return False, ""
    return is_open, ""

if __name__ == "__main__":
    app.run(debug=True)