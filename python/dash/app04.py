import dash
import datetime
from dash import html
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    dbc.Label("Name", style={'min-width': '80px',
                            'margin-right': '10px',
                            'align-self': 'center',
                            'white-space': 'nowrap'}),
    dbc.Textarea(
        id="my-textarea",
        placeholder="Enter the text",
        rows=1,
        cols=50,
        className="mb-3",
        value="This is a text area",
        readOnly=True,
        style={'flex-grow': '1', 'resize': 'none'},
    ),
])

if __name__ == "__main__":
    app.run(debug=True, port=8000)