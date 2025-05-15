import React, { useState } from "react";
import { Box, ChakraProvider } from "@chakra-ui/react";
import Sidebar from "./components/Sidebar";
import ChatArea from "./components/ChatArea";
import axios from "axios";

function App() {
  const [chats, setChats] = useState([]);
  const [currentChat, setCurrentChat] = useState(null);

  const createNewChat = () => {
    const newChat = {
      id: Date.now(),
      name: "새로운 채팅",
      messages: [],
    };
    setChats([...chats, newChat]);
    setCurrentChat(newChat);
  };

  const updateChatName = (chatId, newName) => {
    const updatedChats = chats.map((chat) =>
      chat.id === chatId ? { ...chat, name: newName } : chat
    );
    setChats(updatedChats);
    if (currentChat?.id === chatId) {
      setCurrentChat({ ...currentChat, name: newName });
    }
  };

  const handleSendMessage = async (message) => {
    if (currentChat) {
      const updatedChat = {
        ...currentChat,
        messages: [...currentChat.messages, { content: message, type: "user" }],
      };

      try {
        const response = await axios.post("http://localhost:8000/api/chat", {
          message: message,
        });

        const aiResponse = {
          content: response.data.response,
          type: "assistant",
        };

        updatedChat.messages.push(aiResponse);
        setCurrentChat(updatedChat);
        setChats(
          chats.map((chat) => (chat.id === currentChat.id ? updatedChat : chat))
        );
      } catch (error) {
        console.error("Error:", error);
        const aiResponse = {
          content: "죄송합니다. 일시적인 오류가 발생했습니다.",
          type: "assistant",
        };
        updatedChat.messages.push(aiResponse);
        setCurrentChat(updatedChat);
        setChats(
          chats.map((chat) => (chat.id === currentChat.id ? updatedChat : chat))
        );
      }
    }
  };

  const deleteChat = (chatId) => {
    const updatedChats = chats.filter((chat) => chat.id !== chatId);
    setChats(updatedChats);
    if (currentChat?.id === chatId) {
      setCurrentChat(null);
    }
  };

  return (
    <ChakraProvider>
      <Box display="flex" h="100vh" bg="gray.800">
        <Sidebar
          chats={chats}
          currentChat={currentChat}
          onChatSelect={setCurrentChat}
          onNewChat={createNewChat}
          onDeleteChat={deleteChat}
        />
        <ChatArea
          currentChat={currentChat}
          onSendMessage={handleSendMessage}
          onUpdateName={updateChatName}
        />
      </Box>
    </ChakraProvider>
  );
}

export default App;
