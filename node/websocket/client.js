const WebSocket = require('ws');

const ws = new WebSocket('ws://localhost:8080');

ws.onopen = () => {
    console.log('서버에 연결되었습니다.');
    // 3초마다 메시지 전송
    setInterval(() => {
        const message = `안녕하세요, 클라이언트에서 보낸 메시지입니다. (시간: ${new Date().toLocaleTimeString()})`;
        console.log(`서버로 메시지 전송: ${message}`);
        ws.send(message);
    }, 3000);
};

ws.onmessage = event => {
    try {
        const data = JSON.parse(event.data); // JSON 파싱 시도
        if (data.senderId && data.message) {
            console.log(`서버로부터 메시지 수신: [발신자 ID: ${data.senderId}] ${data.message}`);
        } else {
            console.log(`서버로부터 원시 메시지 수신: ${event.data}`);
        }
    } catch (e) {
        // JSON 파싱 실패 시 일반 텍스트로 처리
        console.log(`서버로부터 원시 메시지 수신 (파싱 오류): ${event.data}`);
    }
};

ws.onclose = () => {
    console.log('서버와의 연결이 해제되었습니다.');
};

ws.onerror = error => {
    console.error(`WebSocket 오류 발생: ${error.message}`);
};
