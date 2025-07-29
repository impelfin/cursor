import dash
from dash import html

app = dash.Dash(__name__)

app.layout = html.Div(
    [
        html.H1("Hello from Dash!"),
        html.P("This is a simple Dash application."),
        html.Button('Click!!', className="btn btn-primary mt-4"),
    ],
    className="container",
)

if __name__ == "__main__":
    app.run(debug=True, port=8000)