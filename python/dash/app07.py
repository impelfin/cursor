import dash
from dash import html
import dash_bootstrap_components as dbc
from dash import Input, Output
import dash_ag_grid as ag
import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Robert', 'James'],
    'Age': [25, 30, 35, 40, 45, 50],
    'City': ['New York', 'San Francisco', 'Los Angeles', 'Chicago', 'Seoul', 'Busan']
}
df = pd.DataFrame(data)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    dbc.Container([
        html.H1('Dash AG Grid Example'),
        dbc.Row([
            dbc.Col([
                ag.AgGrid(
                    id='ag-grid',
                    columnDefs=[
                        {"headerName": "Name", "field": "Name"},
                        {"headerName": "Age", "field": "Age"},
                        {"headerName": "City", "field": "City"},
                    ],
                   dashGridOptions={"rowSelection": "single",
                                    "pagination": True, "paginationPageSize": "50"},                 
                    rowData=df.to_dict('records'),
                    columnSize="sizeToFit",
                    defaultColDef={"resizable": True},
                    style={"height": 300, "width": "100%"}
                ),
            ], md=10),
        ]),
    ]),
])

if __name__ == "__main__":
    app.run(debug=True, port=8000)