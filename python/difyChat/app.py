import requests
import json
import os
# pprint는 이제 사용하지 않으므로 제거합니다.

DIFY_API_KEY = os.getenv("DIFY_API_KEY")
if not DIFY_API_KEY:
    raise ValueError("DIFY_API_KEY 환경 변수가 설정되지 않았습니다.")

DIFY_API_ENDPOINT = 'https://api.dify.ai/v1/chat-messages'

def send_chat_message(query: str, user_id: str = "default_user", conversation_id: str = ""):
    headers = {
        'Authorization': f'Bearer {DIFY_API_KEY}',
        'Content-Type': 'application/json'
    }

    data = {
        "inputs": {
            "user_question": query
        },
        "query": query,
        "response_mode": "streaming",
        "conversation_id": conversation_id,
        "user": user_id,
        "files": []
    }

    # "Sending message to Dify API: {query}" 이 메시지는 사용자에게 보이지 않게 제거합니다.
    try:
        response = requests.post(DIFY_API_ENDPOINT, headers=headers, data=json.dumps(data), stream=True)

        if response.status_code != 200:
            print(f"\n--- Dify API Error Response (Status {response.status_code}) ---")
            print(response.text)
            print("--------------------------------------------------")

        response.raise_for_status()

        full_response_content = ""
        # "AI: " 접두사는 첫 번째 응답 전에 한 번만 출력합니다.
        print("AI: ", end="") 
        
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                try:
                    chunk_str = chunk.decode('utf-8')
                    for line in chunk_str.splitlines():
                        if line.startswith('data: '):
                            json_data_str = line[len('data: '):]
                            event_data = json.loads(json_data_str)
                            
                            if event_data.get('event') == 'agent_message' and 'answer' in event_data:
                                if event_data['answer']:
                                    print(event_data['answer'], end='')
                                    full_response_content += event_data['answer']
                            elif event_data.get('event') == 'message_end':
                                # "--- Dify 응답 종료 ---" 이 메시지는 사용자에게 보이지 않게 제거합니다.
                                print("") # 마지막 응답 후 새 줄 추가
                                return event_data

                except json.JSONDecodeError as e:
                    pass

        print("") # 스트리밍이 message_end 없이 끝날 경우를 대비해 새 줄 추가
        return {"answer": full_response_content}
    except requests.exceptions.RequestException as e:
        print(f"\nAPI 요청 중 오류 발생: {e}")
        return {"error": str(e)}
    except Exception as e:
        print(f"\n예상치 못한 오류 발생: {e}")
        return {"error": str(e)}

def chat_app():
    print("Dify 챗 애플리케이션을 시작합니다. 'exit'을 입력하여 종료하세요.")
    current_conversation_id = ""
    user_id = "your_unique_user_id"

    while True:
        user_input = input("You: ") # 사용자 입력 프롬프트를 "You: "로 변경
        if user_input.lower() == 'exit':
            print("챗 애플리케이션을 종료합니다.")
            break

        response_data = send_chat_message(user_input, user_id, current_conversation_id)

        if "error" in response_data:
            print(f"오류: {response_data['error']}")
        else:
            if response_data.get("conversation_id"):
                current_conversation_id = response_data["conversation_id"]
            elif not current_conversation_id and response_data.get("id"):
                current_conversation_id = response_data["id"]

if __name__ == "__main__":
    chat_app()
