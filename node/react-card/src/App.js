import React, { useState, useEffect, useRef } from 'react';
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
  const [elapsedTime, setElapsedTime] = useState(0);
  const [gameStarted, setGameStarted] = useState(false);
  const timerRef = useRef(null);
  const [isWrongMatch, setIsWrongMatch] = useState(false);
  const [showResultPopup, setShowResultPopup] = useState(false); // 결과 팝업 표시 여부
  const [finalTurns, setFinalTurns] = useState(0); // 최종 턴 수 저장
  const [finalTime, setFinalTime] = useState(0); // 최종 시간 저장

  // 카드 섞기 함수
  const shuffleCards = () => {
    const shuffledCards = [...cardImages, ...cardImages]
      .sort(() => Math.random() - 0.5)
      .map((card) => ({ ...card, id: Math.random() }));

    setChoiceOne(null);
    setChoiceTwo(null);
    setCards(shuffledCards);
    setTurns(0);
    setElapsedTime(0);
    setDisabled(true);
    setIsWrongMatch(false);
    setShowResultPopup(false); // 새 게임 시작 시 팝업 닫기

    // 기존 타이머가 있다면 클리어
    if (timerRef.current) {
      clearInterval(timerRef.current);
    }

    // 게임 시작 시 3초간 카드 보여주기
    setGameStarted(false);
    setTimeout(() => {
      setCards(prevCards => prevCards.map(card => ({ ...card, matched: false })));
      setDisabled(false);
      setGameStarted(true);
      // 타이머 시작
      timerRef.current = setInterval(() => {
        setElapsedTime(prevTime => prevTime + 1);
      }, 1000);
    }, 3000); // 3초 후에 카드 뒤집기
  };

  // 카드 선택 함수
  const handleChoice = (card) => {
    if (!gameStarted || disabled) return;
    if (card.matched) return;
    if (card.id === choiceOne?.id) return;

    choiceOne ? setChoiceTwo(card) : setChoiceOne(card);
  };

  // 두 카드 비교 로직
  useEffect(() => {
    if (choiceOne && choiceTwo) {
      setDisabled(true);
      if (choiceOne.src === choiceTwo.src) {
        // 일치하는 경우
        setCards(prevCards => {
          return prevCards.map(card => {
            if (card.src === choiceOne.src) {
              return { ...card, matched: true }
            } else {
              return card
            }
          })
        });
        setIsWrongMatch(false);
        resetTurn();
      } else {
        // 일치하지 않는 경우
        setIsWrongMatch(true);
        setTimeout(() => {
          setIsWrongMatch(false);
          resetTurn();
        }, 1000);
      }
    }
  }, [choiceOne, choiceTwo]);

  // 모든 카드가 맞춰졌는지 확인 (게임 종료)
  useEffect(() => {
    if (gameStarted && cards.length > 0 && cards.every(card => card.matched)) {
      if (timerRef.current) {
        clearInterval(timerRef.current); // 모든 카드가 맞춰지면 타이머 중지
        timerRef.current = null; // 타이머 참조 초기화
      }
      setFinalTurns(turns); // 최종 턴 수 저장
      setFinalTime(elapsedTime); // 최종 시간 저장
      setShowResultPopup(true); // 결과 팝업 표시
      setGameStarted(false); // 게임 상태 종료로 변경 (선택 불가)
    }
  }, [cards, gameStarted, turns, elapsedTime]);


  // 턴 리셋 및 다음 턴 시작
  const resetTurn = () => {
    setChoiceOne(null);
    setChoiceTwo(null);
    setTurns(prevTurns => prevTurns + 1);
    setDisabled(false);
  };

  // 컴포넌트 마운트 시 카드 섞기 실행
  useEffect(() => {
    shuffleCards();
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, []);

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
        {cards.map(card => {
          const isSelected = card.id === choiceOne?.id || card.id === choiceTwo?.id;
          const showAsFlipped = card.matched || !gameStarted || isSelected;

          return (
            <div
              className={`card ${isWrongMatch && isSelected ? 'wrong-match' : ''}`}
              key={card.id}
              onClick={() => handleChoice(card)}
            >
              <div className={showAsFlipped ? "flipped" : ""}>
                <img
                  className="front"
                  src={card.src}
                  alt="card front"
                  // 맞았을 때 멋지게 오픈되는 효과 추가
                  style={{ transition: card.matched ? 'transform 0.5s ease-out, box-shadow 0.5s ease-out' : '' }}
                />
                <img className="back" src="/img/cover.png" alt="card back" />
              </div>
            </div>
          );
        })}
      </div>
      <p>턴 수: {turns}</p>
      <p>경과 시간: {formatTime(elapsedTime)}</p>

      {/* 게임 결과 팝업 */}
      {showResultPopup && (
        <div className="popup-overlay">
          <div className="popup-content">
            <h2>게임 완료!</h2>
            <p>최종 턴 수: <strong>{finalTurns}</strong></p>
            <p>경과 시간: <strong>{formatTime(finalTime)}</strong></p>
            <button onClick={shuffleCards}>다시 하기</button>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
