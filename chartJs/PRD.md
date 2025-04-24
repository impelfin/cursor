# PRD : chartJs-mvc

## 프로젝트 목표
- models 폴더 아래 json 파일을 chart.js 막대그래프로 시각화
- Node.js 백엔드, 프론트는 HTML/CSS/JS, 포트 8000
- MVC 패턴 적용 (models, controller, public)

## 디렉토리 구조

```
chartJs/
├── app.js                # 서버 구동 및 라우팅
├── package.json          # 의존성 관리
├── project.md            # 기획 및 구조 설명
├── models/               # json 데이터 파일 위치
│   └── pension_calculation.pension.json
├── controller/           # 백엔드 컨트롤러
│   └── chartController.js
└── public/               # 프론트엔드 코드
    ├── index.html
    ├── style.css
    └── chart.js
```

## 기능 요약
- **GET /api/data**: models 폴더의 json 파일을 읽어 데이터 제공
- 프론트엔드에서 fetch로 데이터 받아 chart.js로 막대그래프 렌더링
