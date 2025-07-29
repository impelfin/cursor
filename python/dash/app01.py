import dash
from dash import html
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

app.layout = html.Div(
    children=[
        dbc.Row(
            children=[
                dbc.Col(html.Div('첫번째 열'), width=6),
                dbc.Col(html.Div('두번째 열'), width=6),
            ]
        ),
        dbc.Row(
            children=[
                dbc.Col(html.Div('세번째 열'), width=4),
                dbc.Col(html.Div('네번째 열'), width=4),
                dbc.Col(html.Div('다섯번째 열'), width=4),
            ]
        ),
    ]
)

if __name__ == "__main__":
    app.run(debug=True, port=8000)