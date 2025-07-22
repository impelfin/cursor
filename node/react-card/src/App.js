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
  const [isWrongMatch, setIsWrongMatch] = useState(false); // 틀렸을 때 빨간 테두리 표시용 상태 추가

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
    setDisabled(true); // 초기 노출 동안 카드 선택 비활성화
    setIsWrongMatch(false); // 새 게임 시작 시 경고 상태 초기화

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
    if (card.id === choiceOne?.id) return; // 이미 선택된 카드를 다시 선택하는 것 방지 (id로 비교)

    choiceOne ? setChoiceTwo(card) : setChoiceOne(card);
  };

  // 두 카드 비교 로직
  useEffect(() => {
    if (choiceOne && choiceTwo) {
      setDisabled(true); // 카드 선택 일시 비활성화
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
        setIsWrongMatch(false); // 매칭 성공 시 경고 상태 해제
        resetTurn(); // 바로 다음 턴 시작 (애니메이션은 CSS에서 처리)
      } else {
        // 일치하지 않는 경우
        setIsWrongMatch(true); // 틀렸을 때 경고 상태 활성화
        setTimeout(() => {
          setIsWrongMatch(false); // 1초 후 경고 상태 해제
          resetTurn();
        }, 1000); // 1초 후에 카드 뒤집기
      }
    }
  }, [choiceOne, choiceTwo]);

  // 모든 카드가 맞춰졌는지 확인 (게임 종료)
  useEffect(() => {
    // 카드가 로드된 후 (cards.length > 0) 모든 카드가 matched 상태인지 확인
    if (gameStarted && cards.length > 0 && cards.every(card => card.matched)) {
      if (timerRef.current) {
        clearInterval(timerRef.current); // 모든 카드가 맞춰지면 타이머 중지
      }
      // 선택 사항: 게임 종료 메시지 등 추가 가능
      // alert(`게임 완료! 턴 수: ${turns}, 경과 시간: ${elapsedTime}초`);
    }
  }, [cards, gameStarted, turns, elapsedTime]);


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
          // 카드가 선택되었는지, 틀렸는지, 맞춰졌는지에 따라 클래스 추가
          const isSelected = card.id === choiceOne?.id || card.id === choiceTwo?.id;
          const showAsFlipped = card.matched || !gameStarted || isSelected;

          return (
            <div
              className={`card ${isWrongMatch && isSelected ? 'wrong-match' : ''}`} // 틀렸을 때 경고 클래스 추가
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
    </div>
  );
}

export default App;
