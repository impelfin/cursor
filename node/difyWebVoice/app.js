require('dotenv').config(); // .env 파일에서 환경 변수 로드
const express = require('express');
const path = require('path');
// Node.js 18 이상에서는 fetch가 내장되어 있으므로 node-fetch를 require할 필요가 없습니다.
// 만약 Node.js 버전이 18 미만이거나, 특정 환경에서 오류가 발생한다면
// const fetch = require('node-fetch'); // 이 줄의 주석을 해제하세요.

const app = express();
const port = process.env.PORT || 8000; // 환경 변수 PORT가 없으면 8000번 포트 사용

const DIFY_API_KEY = process.env.DIFY_API_KEY;
if (!DIFY_API_KEY) {
    console.error("DIFY_API_KEY 환경 변수가 설정되지 않았습니다. .env 파일을 확인해주세요.");
    process.exit(1); // API 키가 없으면 서버 시작 중단
}

const DIFY_API_ENDPOINT = 'https://api.dify.ai/v1/chat-messages';

// JSON 요청 본문 파싱을 위한 미들웨어 설정
app.use(express.json());
// `public` 폴더의 정적 파일들(index.html, CSS, JS 등)을 서빙하도록 설정
app.use(express.static(path.join(__dirname, 'public')));

// 간단한 세션 관리를 위한 딕셔너리 (실제 운영 환경에서는 Redis 또는 데이터베이스 사용을 권장)
// 사용자 ID별로 대화 ID를 저장합니다.
const conversationIds = {};

// 기본 라우트: 루트 URL (http://localhost:3000/)로 접속 시 `index.html` 파일 제공
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// `/chat` 엔드포인트: 클라이언트의 질문을 받아 Dify API에 전송하고 스트리밍 응답을 반환
app.post('/chat', async (req, res) => {
    const userQuestion = req.body.question;
    const userId = "web_user_123"; // 간단한 예시를 위해 고정된 사용자 ID 사용

    if (!userQuestion) {
        return res.status(400).json({ error: "No question provided" });
    }

    // 사용자별 현재 대화 ID를 가져오거나, 없으면 빈 문자열로 초기화
    const currentConversationId = conversationIds[userId] || "";

    // 클라이언트로 Server-Sent Events (SSE)를 보내기 위한 헤더 설정
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    // CORS 문제 방지를 위해 필요한 경우 추가 (개발 시 유용)
    res.setHeader('Access-Control-Allow-Origin', '*'); 

    const headers = {
        'Authorization': `Bearer ${DIFY_API_KEY}`,
        'Content-Type': 'application/json'
    };

    const data = {
        "inputs": {
            "user_question": userQuestion // Dify 앱의 입력 폼에 정의된 필드 이름
        },
        "query": userQuestion, // Dify API의 `query` 필드
        "response_mode": "streaming", // Dify로부터 스트리밍 응답 요청
        "conversation_id": currentConversationId, // 대화 연속성을 위한 ID
        "user": userId, // 사용자 ID
        "files": [] // 현재 예시에서는 파일 첨부 사용 안 함
    };

    try {
        // Dify API로 POST 요청 전송
        const difyResponse = await fetch(DIFY_API_ENDPOINT, {
            method: 'POST',
            headers: headers,
            body: JSON.stringify(data),
        });

        // HTTP 응답 상태 코드가 200 OK가 아닌 경우 오류 처리
        if (!difyResponse.ok) {
            const errorText = await difyResponse.text();
            console.error(`Dify API Error: ${difyResponse.status} - ${errorText}`);
            // 클라이언트로 오류 메시지를 SSE 이벤트로 전송
            res.write(`data: ${JSON.stringify({ type: 'error', content: `Dify API 오류: ${difyResponse.status} - ${errorText.substring(0, Math.min(errorText.length, 100))}...` })}\n\n`);
            return res.end(); // 스트림 종료
        }

        // Dify의 스트리밍 응답을 읽기 위한 Reader와 Decoder 설정
        const reader = difyResponse.body.getReader();
        const decoder = new TextDecoder('utf-8');
        let buffer = ''; // 불완전한 라인을 저장할 버퍼

        // Dify 스트림을 계속 읽어와 클라이언트로 SSE 이벤트로 전달
        while (true) {
            const { done, value } = await reader.read();
            if (done) {
                // 스트림이 끝나면 남은 버퍼 처리 (마지막 라인이 불완전할 수 있음)
                // console.log("Stream ended, remaining buffer:", buffer); // 디버깅용
                break;
            }

            const chunk = decoder.decode(value, { stream: true });
            buffer += chunk; // 현재 청크를 버퍼에 추가

            // 버퍼에서 완전한 줄을 찾습니다.
            // 각 'data:' 라인은 '\n'으로 끝나고, 연속적인 '\n\n'은 이벤트의 끝을 나타냅니다.
            let lastNewlineIndex = buffer.lastIndexOf('\n');
            if (lastNewlineIndex > -1) {
                const completeLines = buffer.substring(0, lastNewlineIndex + 1); // 완전한 라인들
                buffer = buffer.substring(lastNewlineIndex + 1); // 다음을 위해 남은 불완전한 부분

                for (const line of completeLines.split('\n')) {
                    if (line.startsWith('data: ')) {
                        try {
                            const jsonString = line.substring(6); // 'data: ' 제거
                            if (jsonString.trim() === '') continue; // 빈 데이터 라인 스킵

                            const eventData = JSON.parse(jsonString);
                            
                            if (eventData.event === 'agent_message' && eventData.answer) {
                                res.write(`data: ${JSON.stringify({ type: 'text', content: eventData.answer })}\n\n`);
                            } else if (eventData.event === 'message_end') {
                                const newConvId = eventData.conversation_id || eventData.id;
                                if (newConvId) {
                                    conversationIds[userId] = newConvId;
                                }
                                res.write(`data: ${JSON.stringify({ type: 'end' })}\n\n`);
                                res.end();
                                return;
                            }
                        } catch (e) {
                            console.warn("Failed to parse Dify chunk line (JSON error):", line, e.message); // 오류 메시지 간결화
                        }
                    } else if (line.trim() !== '') {
                        // 'data:'로 시작하지 않지만 비어있지 않은 라인은 경고
                        console.warn("Unexpected line in Dify stream (not data:):", line);
                    }
                }
            }
        }
        // 스트림이 끝났는데 message_end 이벤트가 오지 않은 경우 대비
        res.write(`data: ${JSON.stringify({ type: 'end' })}\n\n`);
        res.end();

    } catch (error) {
        console.error("서버 내부 오류:", error);
        res.write(`data: ${JSON.stringify({ type: 'error', content: `서버 내부 오류: ${error.message}` })}\n\n`);
        res.end();
    }
});

// Express 서버 시작
app.listen(port, () => {
    console.log(`Express Dify Chat Server listening at http://localhost:${port}`);
    console.log(`Dify API Key: ${DIFY_API_KEY ? 'Loaded' : 'NOT LOADED - Check .env file!'}`);
});
