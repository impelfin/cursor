const WebSocket = require('ws');
const { v4: uuidv4 } = require('uuid'); // 고유 ID 생성을 위한 라이브러리

const wss = new WebSocket.Server({ port: 8080 });

console.log('WebSocket 서버가 8080 포트에서 실행 중입니다.');

wss.on('connection', ws => {
    // 클라이언트 연결 시 고유 ID 부여
    ws.id = uuidv4();
    console.log(`클라이언트가 연결되었습니다. ID: ${ws.id}`);

    ws.on('message', message => {
        const receivedMessage = message.toString();
        // 메시지를 보낸 클라이언트의 ID와 함께 로그 출력
        console.log(`클라이언트 ${ws.id} 로부터 수신된 메시지: ${receivedMessage}`);

        // 예시: 특정 클라이언트에게만 응답 (ID가 'abc-123'인 클라이언트에게만)
        if (receivedMessage.includes('프라이빗')) {
            wss.clients.forEach(client => {
                if (client.id === 'YOUR_TARGET_CLIENT_ID' && client.readyState === WebSocket.OPEN) {
                    client.send(`[${ws.id} 로부터의 프라이빗 메시지]: ${receivedMessage}`);
                }
            });
            return; // 특정 클라이언트에게 보냈으므로 여기서 처리 종료
        }

        // 예시: 모든 연결된 클라이언트에게 메시지 브로드캐스트 (발신자 ID 포함)
        wss.clients.forEach(client => {
            if (client.readyState === WebSocket.OPEN) {
                // 발신자 ID와 메시지를 함께 보냄
                client.send(JSON.stringify({
                    senderId: ws.id,
                    message: receivedMessage
                }));
            }
        });
    });

    ws.on('close', () => {
        console.log(`클라이언트 ${ws.id} 연결이 해제되었습니다.`);
    });

    ws.on('error', error => {
        console.error(`클라이언트 ${ws.id} 에서 WebSocket 오류 발생: ${error.message}`);
    });
});
