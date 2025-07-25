## PRD: 카드 뒤집기 게임

### 1. 개요 (Introduction)

본 문서는 간단한 웹 기반 카드 뒤집기(메모리) 게임의 제품 요구사항을 설명합니다. 사용자는 여러 장의 카드 중 동일한 이미지를 가진 두 장의 카드를 찾아 짝을 맞추는 것이 목표입니다. 이 게임은 사용자의 기억력과 집중력을 향상시키는 데 중점을 둡니다.

### 2. 목표 (Goals)

* 사용자에게 간단하고 즐거운 메모리 게임 경험 제공
* React 프레임워크를 활용하여 웹 기반 게임 구현 능력 시연
* 직관적인 사용자 인터페이스를 통해 모든 연령대의 사용자가 쉽게 접근하고 즐길 수 있도록 함
* 사용자의 턴 수를 기록하여 게임 진행 상황 피드백 제공

### 3. 기능 요구사항 (Functional Requirements)

* **FR1. 게임 초기화 및 시작**:
    * **FR1.1** 게임 시작 시 모든 카드를 섞어 뒷면이 보이도록 배치해야 합니다.
    * **FR1.2** "새 게임 시작" 버튼을 클릭하면 현재 게임 상태를 초기화하고 새로운 카드를 섞어 배치해야 합니다.
    * **FR1.3** 컴포넌트 마운트 시 자동으로 게임이 시작되어야 합니다.
* **FR2. 카드 선택**:
    * **FR2.1** 사용자는 한 번에 두 장의 카드를 선택할 수 있어야 합니다.
    * **FR2.2** 카드를 클릭하면 해당 카드의 앞면(이미지)이 보여야 합니다.
    * **FR2.3** 이미 뒤집혀 있거나(앞면이 보이는 상태), 짝이 맞춰진 카드는 다시 선택할 수 없어야 합니다.
    * **FR2.4** 두 장의 카드가 선택되어 비교 중일 때는 다른 카드를 선택할 수 없도록 입력이 비활성화되어야 합니다.
* **FR3. 카드 비교 및 일치**:
    * **FR3.1** 두 장의 카드가 선택되면, 두 카드의 이미지가 서로 같은지 비교해야 합니다.
    * **FR3.2** 이미지가 같을 경우, 두 카드는 앞면이 보이도록 유지되어야 하며, 다시 뒤집을 수 없도록 '짝 맞춰짐' 상태로 변경되어야 합니다.
    * **FR3.3** 이미지가 다를 경우, 약 1초 후에 두 카드는 자동으로 다시 뒷면이 보이도록 뒤집혀야 합니다.
* **FR4. 턴 수 계산**:
    * **FR4.1** 사용자가 두 장의 카드를 선택하고 비교하는 과정이 한 턴으로 계산되어야 합니다.
    * **FR4.2** 총 턴 수가 화면에 실시간으로 표시되어야 합니다.
    * **FR4.3** 새 게임 시작 시 턴 수는 0으로 초기화되어야 합니다.
* **FR5. 게임 종료 (암묵적)**:
    * **FR5.1** 모든 카드가 짝을 맞춰 앞면이 보이도록 뒤집혔을 때 게임이 완료된 것으로 간주합니다. (현재 구현에서는 명시적인 게임 종료 메시지나 로직 없음)

### 4. 비기능 요구사항 (Non-Functional Requirements)

* **NFR1. 성능**:
    * **NFR1.1** 카드 뒤집기 및 비교 애니메이션은 부드럽게 작동해야 하며, 지연 없이 반응해야 합니다.
* **NFR2. 사용성 (Usability)**:
    * **NFR2.1** UI는 직관적이고 이해하기 쉬워야 합니다.
    * **NFR2.2** 버튼과 카드 클릭 영역은 충분히 커서 사용자가 쉽게 상호작용할 수 있어야 합니다.
* **NFR3. 기술 스택**:
    * **NFR3.1** React.js (Hooks 포함)를 사용하여 개발되어야 합니다.
    * **NFR3.2** CSS를 사용하여 스타일링되어야 합니다.
* **NFR4. 확장성 (Scalability)**:
    * **NFR4.1** 향후 카드 이미지의 종류를 쉽게 추가하거나 변경할 수 있도록 구조화되어야 합니다.

### 5. 사용자 인터페이스 (UI) 요구사항

* **UI1. 게임 보드**:
    * **UI1.1** 카드들은 균일한 그리드 형태로 배열되어야 합니다.
    * **UI1.2** 각 카드는 앞면(이미지)과 뒷면(커버 이미지)을 명확하게 구분하여 표시해야 합니다.
* **UI2. 상태 표시**:
    * **UI2.1** 현재 턴 수가 화면 상단 또는 하단에 명확하게 표시되어야 합니다.
* **UI3. 버튼**:
    * **UI3.1** "새 게임 시작" 버튼이 명확하게 표시되어야 합니다.

### 6. 파일 및 디렉토리 구조 (File and Directory Structure)

/
├── public/
│   ├── index.html
│   └── img/
│       ├── cover.png         # 카드 뒷면 이미지
│       ├── helmet-1.png
│       ├── potion-1.png
│       ├── ring-1.png
│       ├── scroll-1.png
│       ├── shield-1.png
│       └── sword-1.png       # 카드 앞면 이미지들
├── src/
│   ├── App.js                # 메인 React 컴포넌트 (게임 로직 및 UI)
│   ├── App.css               # App 컴포넌트 스타일 시트
│   ├── index.js              # React 앱 진입점
│   └── index.css             # 전역 스타일 또는 기본 스타일
├── .gitignore
├── package.json
├── package-lock.json
└── README.md

* **`public/`**: 정적 자산(Static Assets)을 포함하는 디렉토리입니다.
    * **`index.html`**: React 앱이 마운트되는 기본 HTML 파일입니다.
    * **`img/`**: 카드 게임에 사용되는 모든 이미지 파일을 저장하는 디렉토리입니다. `cover.png`는 카드 뒷면 이미지로, 나머지 파일들은 카드 앞면 이미지로 사용됩니다.
* **`src/`**: React 애플리케이션의 소스 코드 디렉토리입니다.
    * **`App.js`**: 게임의 핵심 로직과 UI를 담당하는 메인 React 컴포넌트입니다. `useState`, `useEffect` 훅을 사용하여 게임 상태를 관리하고 렌더링합니다.
    * **`App.css`**: `App.js` 컴포넌트의 스타일을 정의하는 CSS 파일입니다. 카드 그리드, 카드 뒤집기 애니메이션 등을 포함합니다.
    * **`index.js`**: React 애플리케이션의 진입점(Entry Point)입니다. `ReactDOM.render()`를 사용하여 `App` 컴포넌트를 `index.html`에 렌더링합니다.
    * **`index.css`**: 전체 애플리케이션에 적용되는 전역 스타일 또는 기본 스타일을 정의하는 CSS 파일입니다.
* **`.gitignore`**: Git 버전 관리에서 제외할 파일 및 디렉토리를 지정합니다. (예: `node_modules`, 빌드 결과물)
* **`package.json`**: 프로젝트의 메타데이터, 의존성(dependencies), 스크립트(scripts)를 정의하는 파일입니다.
* **`package-lock.json`**: `package.json`에 명시된 의존성들의 정확한 버전을 기록하여 빌드 일관성을 보장합니다.
* **`README.md`**: 프로젝트에 대한 설명, 설치 및 실행 방법 등을 담는 문서 파일입니다.

### 7. 향후 개선 사항 (Future Enhancements)

* **FE1. 게임 종료 메시지**: 모든 카드를 맞췄을 때 "게임 완료!" 메시지와 함께 최종 턴 수를 보여주는 기능 추가.
* **FE2. 최고 기록**: 플레이어의 최고 턴 수를 기록하고 표시하는 기능 추가.
* **FE3. 난이도 조절**: 카드 이미지의 개수를 조절하여 게임 난이도를 변경하는 기능 추가.
* **FE4. 타이머**: 게임 진행 시간을 측정하고 표시하는 기능 추가.
* **FE5. 반응형 디자인**: 다양한 화면 크기(모바일, 태블릿, 데스크톱)에 맞춰 레이아웃이 유동적으로 변경되도록 개선.
