import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output
from styles import *

app = Dash(__name__)

df = pd.read_csv("data/intro_bees.csv")
df = df.groupby(['State', 'ANSI', 'Affected by', 'Year', 'state_code'])[['Pct of Colonies Impacted']].mean()
df.reset_index(inplace=True)
df.head()

# App layout
app.layout = html.Div([
    # Google Fonts 추가
    html.Link(
        rel='stylesheet',
        href=GOOGLE_FONTS_URL
    ),
    
    html.Div([
        html.H1("🐝 Bee Colony Impact Dashboard", style=TITLE_STYLE), 
        
        html.Div([
            html.Label("Select Year:", style=LABEL_STYLE),
            dcc.Dropdown(
                id="slct_year",
                options=[
                    {"label": "2015", "value": 2015},
                    {"label": "2016", "value": 2016},
                    {"label": "2017", "value": 2017},
                    {"label": "2018", "value": 2018}
                ],
                multi=False, 
                value=2015, 
                style=DROPDOWN_STYLE
            )
        ], style=DROPDOWN_CONTAINER_STYLE),
        
        html.Div([
            html.Div(id='output_container', children=[], style=OUTPUT_CONTAINER_STYLE)
        ], style=OUTPUT_CONTAINER_WRAPPER_STYLE),
        
        dcc.Graph(id='my_bee_map', figure={}, style=GRAPH_STYLE)
    ], style=MAIN_CONTAINER_STYLE)
], style=GLOBAL_FONT_STYLE)

@app.callback(
    [Output(component_id='output_container', component_property='children'),
    Output(component_id='my_bee_map', component_property='figure')],
    [Input(component_id='slct_year', component_property='value')]
)
def update_graph(option_slctd):
    print(option_slctd)
    print(type(option_slctd))

    container = "The year chosen by the user is: {}".format(option_slctd)

    dff = df.copy()
    dff = df[dff['Year'] == option_slctd]
    dff = dff[dff['Affected by'] == 'Varroa_mites']
    
    fig = px.choropleth(
        data_frame=dff,
        locationmode='USA-states',
        locations='state_code',
        scope='usa',
        color='Pct of Colonies Impacted',
        hover_data=['State', 'Pct of Colonies Impacted'],
        color_continuous_scale=px.colors.sequential.YlOrRd,
        labels={'Pct of Colonies Impacted': '% of Bee Colonies'},
        template='plotly_dark'
    )

    return container, fig

if __name__ == "__main__":
    app.run(debug=True, port=8000)
