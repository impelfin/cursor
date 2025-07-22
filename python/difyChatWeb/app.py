import os
import json
import requests
from flask import Flask, render_template, request, Response
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드 (DIFY_API_KEY 등)
load_dotenv()

app = Flask(__name__)

# Dify API 키 로드 및 유효성 검사
DIFY_API_KEY = os.getenv("DIFY_API_KEY")
if not DIFY_API_KEY:
    raise ValueError("DIFY_API_KEY 환경 변수가 설정되지 않았습니다. .env 파일을 확인하거나 직접 설정해주세요.")

# Dify API 엔드포인트 URL
# 클라우드 Dify를 사용하고 있다면 이 URL이 맞습니다.
DIFY_API_ENDPOINT = 'https://api.dify.ai/v1/chat-messages'

# 세션 관리를 위한 딕셔너리 (간단한 예시, 실제 운영 환경에서는 데이터베이스/Redis 사용 권장)
# 각 사용자(여기서는 'web_user_123')별로 대화 ID를 저장합니다.
conversation_ids = {}

@app.route('/')
def index():
    """
    웹 애플리케이션의 메인 페이지를 렌더링합니다.
    """
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """
    클라이언트로부터 받은 질문을 Dify API에 전송하고,
    Dify의 스트리밍 응답을 클라이언트로 전달합니다 (Server-Sent Events 방식).
    """
    user_question = request.json.get('question')
    user_id = "web_user_123" # 간단한 예시를 위해 고정된 사용자 ID 사용

    if not user_question:
        return json.dumps({"error": "No question provided"}), 400

    # 사용자별 현재 대화 ID를 가져옵니다. 없으면 빈 문자열입니다.
    current_conversation_id = conversation_ids.get(user_id, "")

    headers = {
        'Authorization': f'Bearer {DIFY_API_KEY}',
        'Content-Type': 'application/json'
    }

    data = {
        "inputs": {
            "user_question": user_question # Dify 앱의 "Question" 필드 이름에 맞게 수정
        },
        "query": user_question,
        "response_mode": "streaming", # Dify로부터 스트리밍 응답을 요청
        "conversation_id": current_conversation_id, # 대화의 연속성을 위해 ID 포함
        "user": user_id,
        "files": [] # 현재 예시에서는 파일 첨부 사용 안 함
    }

    def generate():
        """
        Dify API로부터 스트리밍 응답을 받아와 클라이언트로 실시간으로 yield 합니다.
        Server-Sent Events (SSE) 형식으로 데이터를 보냅니다.
        """
        try:
            response = requests.post(DIFY_API_ENDPOINT, headers=headers, data=json.dumps(data), stream=True)
            response.raise_for_status() # HTTP 오류 (예: 4xx, 5xx) 발생 시 예외 발생

            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    try:
                        chunk_str = chunk.decode('utf-8')
                        # 각 라인을 파싱하여 Dify의 SSE 이벤트 데이터를 처리
                        for line in chunk_str.splitlines():
                            if line.startswith('data: '):
                                event_data = json.loads(line[len('data: '):])

                                if event_data.get('event') == 'agent_message' and 'answer' in event_data:
                                    answer_text = event_data['answer']
                                    if answer_text:
                                        # 클라이언트로 답변 청크를 SSE 이벤트로 전송
                                        yield f"data: {json.dumps({'type': 'text', 'content': answer_text})}\n\n"
                                elif event_data.get('event') == 'message_end':
                                    # 최종 대화 ID를 업데이트하여 다음 질문에 사용
                                    new_conv_id = event_data.get("conversation_id") or event_data.get("id")
                                    if new_conv_id:
                                        conversation_ids[user_id] = new_conv_id
                                    # 클라이언트에게 스트리밍 종료를 알림
                                    yield f"data: {json.dumps({'type': 'end'})}\n\n"
                                    return # 스트림 종료

                    except json.JSONDecodeError:
                        # 불완전하거나 유효하지 않은 JSON 청크는 무시
                        pass

        except requests.exceptions.RequestException as e:
            # API 요청 중 발생한 네트워크 또는 HTTP 오류 처리
            yield f"data: {json.dumps({'type': 'error', 'content': f'API 요청 중 오류 발생: {e}'})}\n\n"
        except Exception as e:
            # 예상치 못한 기타 오류 처리
            yield f"data: {json.dumps({'type': 'error', 'content': f'예상치 못한 오류 발생: {e}'})}\n\n"

    # Server-Sent Events (SSE) 형식으로 응답 반환
    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    # Flask 앱 실행
    # debug=True는 개발 중에는 편리하지만, 실제 운영 환경에서는 False로 설정해야 합니다.
    app.run(debug=True)
