import React, { useState, useEffect, useRef } from 'react'; // useRef 추가
import './App.css';

const cardImages = [
  { "src": "/img/helmet-1.png", matched: false },
  { "src": "/img/potion-1.png", matched: false },
  { "src": "/img/ring-1.png", matched: false },
  { "src": "/img/scroll-1.png", matched: false },
  { "src": "/img/shield-1.png", matched: false },
  { "src": "/img/sword-1.png", matched: false }
];

function App() {
  const [cards, setCards] = useState([]);
  const [turns, setTurns] = useState(0);
  const [choiceOne, setChoiceOne] = useState(null);
  const [choiceTwo, setChoiceTwo] = useState(null);
  const [disabled, setDisabled] = useState(false);
  const [elapsedTime, setElapsedTime] = useState(0); // 경과 시간 상태 추가
  const [gameStarted, setGameStarted] = useState(false); // 게임 시작 여부 상태 추가
  const timerRef = useRef(null); // 타이머 ID를 저장할 ref 추가

  // 카드 섞기 함수
  const shuffleCards = () => {
    const shuffledCards = [...cardImages, ...cardImages]
      .sort(() => Math.random() - 0.5)
      .map((card) => ({ ...card, id: Math.random() }));

    setChoiceOne(null);
    setChoiceTwo(null);
    setCards(shuffledCards);
    setTurns(0);
    setElapsedTime(0); // 시간 초기화
    setDisabled(true); // 초기 노출 동안 카드 선택 비활성화

    // 기존 타이머가 있다면 클리어
    if (timerRef.current) {
      clearInterval(timerRef.current);
    }

    // 게임 시작 시 3초간 카드 보여주기
    setGameStarted(false); // 게임 시작 전 상태로 설정하여 모든 카드 보이게 함
    setTimeout(() => {
      setCards(prevCards => prevCards.map(card => ({ ...card, matched: false }))); //matched를 false로 설정하여 다시 뒤집히게 함
      setDisabled(false); // 카드 선택 활성화
      setGameStarted(true); // 게임 시작 상태로 변경
      // 타이머 시작
      timerRef.current = setInterval(() => {
        setElapsedTime(prevTime => prevTime + 1);
      }, 1000);
    }, 3000); // 3초 후에 카드 뒤집기
  };

  // 카드 선택 함수
  const handleChoice = (card) => {
    if (!gameStarted || disabled) return; // 게임이 시작되지 않았거나 비활성화 상태면 클릭 무시
    if (card.matched) return; // 이미 맞춰진 카드는 클릭 무시
    if (card === choiceOne) return; // 같은 카드를 두 번 클릭하는 것 방지

    choiceOne ? setChoiceTwo(card) : setChoiceOne(card);
  };

  // 두 카드 비교 로직
  useEffect(() => {
    if (choiceOne && choiceTwo) {
      setDisabled(true); // 카드 선택 일시 비활성화
      if (choiceOne.src === choiceTwo.src) {
        setCards(prevCards => {
          return prevCards.map(card => {
            if (card.src === choiceOne.src) {
              return { ...card, matched: true }
            } else {
              return card
            }
          })
        })
        resetTurn();
      } else {
        setTimeout(() => resetTurn(), 1000);
      }
    }
  }, [choiceOne, choiceTwo]);

  // 모든 카드가 맞춰졌는지 확인 (게임 종료)
  useEffect(() => {
    if (cards.length > 0 && cards.every(card => card.matched)) {
      if (timerRef.current) {
        clearInterval(timerRef.current); // 모든 카드가 맞춰지면 타이머 중지
      }
      // 선택 사항: 게임 종료 메시지 등 추가 가능
      // alert(`게임 완료! 턴 수: ${turns}, 경과 시간: ${elapsedTime}초`);
    }
  }, [cards, turns, elapsedTime]);


  // 턴 리셋 및 다음 턴 시작
  const resetTurn = () => {
    setChoiceOne(null);
    setChoiceTwo(null);
    setTurns(prevTurns => prevTurns + 1);
    setDisabled(false); // 카드 선택 재활성화
  };

  // 컴포넌트 마운트 시 카드 섞기 실행
  useEffect(() => {
    shuffleCards();
    // 컴포넌트 언마운트 시 타이머 정리
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, []); // [] 의존성 배열로 마운트 시 한 번만 실행

  // 경과 시간을 분:초 형식으로 변환
  const formatTime = (seconds) => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
  };


  return (
    <div className="App">
      <h1>카드 뒤집기 게임</h1>
      <button onClick={shuffleCards}>새 게임 시작</button>

      <div className="card-grid">
        {cards.map(card => (
          <div
            className="card"
            key={card.id}
            onClick={() => handleChoice(card)} // disabled 체크는 handleChoice 내부에서 처리
          >
            {/* gameStarted가 false일 때(초기 노출), 모든 카드를 flipped 상태로 만듭니다. */}
            {/* card.matched는 이미 맞춰진 카드를 계속 flipped 상태로 유지합니다. */}
            <div className={card.matched || !gameStarted ? "flipped" : ""}>
              <img className="front" src={card.src} alt="card front" />
              <img className="back" src="/img/cover.png" alt="card back" />
            </div>
          </div>
        ))}
      </div>
      <p>턴 수: {turns}</p>
      <p>경과 시간: {formatTime(elapsedTime)}</p> {/* 경과 시간 표시 */}
    </div>
  );
}

export default App;
