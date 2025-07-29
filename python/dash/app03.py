import dash
from dash import html
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(
    [
        html.H1("Hello from Dash Bootstrap!"),
        html.P("This is a simple Dash application."),
        dbc.Button('Click!!', color='primary', className="mt-4"),
    ],
    fluid=True,
)

if __name__ == "__main__":
    app.run(debug=True, port=8000)