import dash
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash import callback_context 

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# 앱 레이아웃 정의
app.layout = html.Div([
    html.H2("Dash 모달 예제"),
    dbc.Button("모달 열기", id="modal_edit_open", className="mb-3"), 

    dbc.Modal([
        dbc.ModalHeader("표준셋 추가하기"),
        dbc.ModalBody([
            html.Div([
                dbc.Alert(
                    "질문과 답변을 모두 입력해주세요.",
                    id="edit_warning_alert",
                    color="danger",
                    is_open=False, 
                    dismissable=True
                )
            ]),
            # 질문 입력 필드 추가
            dbc.Label("질문:"),
            dbc.Input(
                id="modal_edit_question",
                type="text",
                placeholder="질문을 입력하세요...",
                className="mb-3"
            ),
            # 답변 입력 필드 추가
            dbc.Label("답변:"),
            dbc.Input(
                id="modal_edit_answer",
                type="text",
                placeholder="답변을 입력하세요...",
                className="mb-3"
            )
        ]),
        dbc.ModalFooter([
            dbc.Button("수정하기", id="edit_standard_set", className="me-2"), 
            dbc.Button("닫기", id="edit_close", className="ms-auto") 
        ])
    ], id="modal_edit", size="lg", is_open=False)
])

@app.callback(
    Output('modal_edit', 'is_open'),
    [Input('modal_edit_open', 'n_clicks'),
     Input('edit_close', 'n_clicks'),
     Input('edit_standard_set', 'n_clicks')], 
    [State('modal_edit', 'is_open'),
     State('modal_edit_question', 'invalid'), 
     State('modal_edit_answer', 'invalid')] 
)
def toggle_modal(n_open, n_close, n_edit, is_open, question_invalid, answer_invalid):
    triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'modal_edit_open' and n_open:
        return not is_open
    elif triggered_id == 'edit_close' and n_close:
        return not is_open
    elif triggered_id == 'edit_standard_set' and n_edit:
        if not question_invalid and not answer_invalid:
             return False 
        return is_open 

    return is_open 

@app.callback(
    [Output('modal_edit_question', 'invalid'),
     Output('modal_edit_answer', 'invalid'),
     Output('edit_warning_alert', 'is_open')],
    [Input('edit_standard_set', 'n_clicks'),
     Input('modal_edit_open', 'n_clicks')], 
    [State('modal_edit_question', 'value'),
     State('modal_edit_answer', 'value')]
)
def update_modal_edit(n_edit, n_open, question, answer):
    triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'edit_standard_set':
        question_invalid = not (question and question.strip()) 
        answer_invalid = not (answer and answer.strip())    
        show_warning = question_invalid or answer_invalid    

        return question_invalid, answer_invalid, show_warning

    elif triggered_id == 'modal_edit_open':
        return False, False, False

    return dash.no_update, dash.no_update, dash.no_update 

if __name__ == "__main__":
    app.run(debug=True, port=8000)
