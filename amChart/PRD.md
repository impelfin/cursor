# PRD: amCharts MVC Web Application

## 목적
- 미국 및 주별 인구 데이터를 amCharts로 시각화하는 웹 애플리케이션을 구축한다.
- 데이터와 프론트엔드 로직을 분리하고, Node.js(Express) 백엔드와 HTML/CSS/JS 프론트엔드로 구성한다.
- MVC(Model-View-Controller) 패턴을 적용하여 구조를 단순하고 명확하게 유지한다.

## 주요 요구사항

1. **백엔드(Node.js/Express)**
    - 8000번 포트에서 서버 구동
    - 정적 파일(public) 서빙
    - `/api/us` : 미국 전체 인구 데이터(usData.json) 반환
    - `/api/state` : 주별 인구 데이터(stateData.json) 반환

2. **프론트엔드(HTML/CSS/JS)**
    - index.html에 amCharts로 인구 피라미드 시각화
    - JS에서 Fetch API로 백엔드 데이터 요청
    - 스타일은 심플하고 직관적으로 구성

3. **데이터**
    - `models/usData.json`, `models/stateData.json`에 JSON 형식으로 저장

4. **구조(MVC)**
    - `models/` : 데이터 파일
    - `controllers/` : 데이터 반환 로직
    - `public/` : 프론트엔드 파일(index.html, main.js, style.css)
    - `app.js` : 서버 및 라우팅

5. **확장성**
    - 주별 데이터 차트 등 기능 확장 용이

## 폴더/파일 구조
```
amChart/
├── public/
│   ├── index.html
│   ├── style.css
│   └── main.js
├── models/
│   ├── usData.json
│   └── stateData.json
├── controllers/
│   └── chartController.js
├── app.js
└── package.json
```

## 사용자 시나리오
1. 서버 실행(`node app.js`)
2. 브라우저에서 `http://localhost:8000` 접속


## 비고

