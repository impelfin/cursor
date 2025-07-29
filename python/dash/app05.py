import dash
import datetime
from dash import html
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

current_time = datetime.datetime.now().strftime('%H:%M:%S')

app.layout = html.Div([
    html.Div("Current Time : " + current_time),
    html.Br(),
    html.Br(),
    dbc.Button("Open Modal", id="open"),
    dbc.Modal([
        dbc.ModalHeader("Header"),
        dbc.ModalBody("This is a content of the modal"),
        dbc.ModalFooter(
            dbc.Button("Close", id="close", className="ml-auto")
        ),
    ], id="modal"),
])

@app.callback(
    Output("modal", "is_open"),
    [Input("open", "n_clicks"), Input("close", "n_clicks")],
    [State("modal", "is_open")],
)
def toggle_modal(open_clicks, close_clicks, is_open):
    if open_clicks or close_clicks:
        return not is_open
    return is_open

if __name__ == "__main__":
    app.run(debug=True, port=8000)