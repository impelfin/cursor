import React, { useState, useEffect } from 'react';
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

  // 카드 섞기 함수
  const shuffleCards = () => {
    const shuffledCards = [...cardImages, ...cardImages]
      .sort(() => Math.random() - 0.5)
      .map((card) => ({ ...card, id: Math.random() }));

    setChoiceOne(null);
    setChoiceTwo(null);
    setCards(shuffledCards);
    setTurns(0);
  };

  // 카드 선택 함수
  const handleChoice = (card) => {
    choiceOne ? setChoiceTwo(card) : setChoiceOne(card);
  };

  // 두 카드 비교 로직
  useEffect(() => {
    if (choiceOne && choiceTwo) {
      setDisabled(true);
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

  // 턴 리셋 및 다음 턴 시작
  const resetTurn = () => {
    setChoiceOne(null);
    setChoiceTwo(null);
    setTurns(prevTurns => prevTurns + 1);
    setDisabled(false);
  };

  // 컴포넌트 마운트 시 카드 섞기 실행
  useEffect(() => {
    shuffleCards()
  }, [])


  return (
    <div className="App">
      <h1>카드 뒤집기 게임</h1>
      <button onClick={shuffleCards}>새 게임 시작</button>

      <div className="card-grid">
        {cards.map(card => (
          <div
            className="card"
            key={card.id}
            onClick={() => !disabled && handleChoice(card)}
          >
            <div className={card.matched ? "flipped" : ""}>
              <img className="front" src={card.src} alt="card front" />
              <img className="back" src="/img/cover.png" alt="card back" />
            </div>
          </div>
        ))}
      </div>
      <p>턴 수: {turns}</p>
    </div>
  );
}

export default App;
