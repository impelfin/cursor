import React, { useState } from "react";
import axios from "axios";

function App() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    // API 통신 로직 구현 예정
  };

  return (
    <div className="App">
      <h1>ChatGPT Clone</h1>
      {/* 채팅 UI 구현 예정 */}
    </div>
  );
}

export default App;
