import dash
from dash import dcc, html
from dash.dependencies import Input, Output

app = dash.Dash(__name__)

app.layout = html.Div(
    [
        dcc.Input(
            id='new-fruit-input',
            type='text',
            placeholder='새로운 과일 입력',
            value=''
        ),
        dcc.Dropdown(
            options=[
                {'label': '사과', 'value': 'apple'},
                {'label': '바나나', 'value': 'banana'},
                {'label': '체리', 'value': 'cherry'}
            ],
            value='apple',
            id='dropdown'
        ),
        dcc.Graph(
            id='graph',
            figure={
                'data': [
                    {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': '사과'},
                ],
                'layout': {
                    'title': '과일의 맛'
                }
            }
        )
    ]
)

@app.callback(
    Output('graph', 'figure'),
    [Input('new-fruit-input', 'value'),
     Input('dropdown', 'value')]
)
def update_graph(new_fruit, selected_fruit):
    data = [
        {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'bar', 'name': selected_fruit},
    ]

    if new_fruit:
        data.append({'x': [1, 2, 3], 'y': [3, 2, 4], 'type': 'bar', 'name': new_fruit})

    return {
        'data': data,
        'layout': {
            'title': '과일의 맛'
        }
    }

if __name__ == "__main__":
    app.run(debug=True, port=8000)