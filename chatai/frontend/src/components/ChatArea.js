import React, { useState, useEffect } from "react";
import { Box, Input, Button, VStack, HStack, Text } from "@chakra-ui/react";
import jsPDF from "jspdf";
import "jspdf-autotable";
import html2canvas from "html2canvas";

function ChatArea({ currentChat, onSendMessage, onUpdateName }) {
  const [message, setMessage] = useState("");
  const [isEditing, setIsEditing] = useState(false);
  const [chatName, setChatName] = useState("");

  useEffect(() => {
    if (currentChat) {
      setChatName(currentChat.name);
    }
  }, [currentChat]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (message.trim()) {
      onSendMessage(message);
      setMessage("");
    }
  };

  const handleNameSubmit = (e) => {
    e.preventDefault();
    if (chatName.trim()) {
      onUpdateName(currentChat.id, chatName);
      setIsEditing(false);
    }
  };

  const handleDownloadPDF = async () => {
    if (!currentChat || !currentChat.messages.length) return;

    const chatContent = document.createElement("div");
    chatContent.style.padding = "20px";

    const titleDiv = document.createElement("div");
    titleDiv.style.textAlign = "center";
    titleDiv.style.fontSize = "24px";
    titleDiv.style.color = "#4299E1";
    titleDiv.style.marginBottom = "20px";
    titleDiv.style.fontFamily =
      "-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif";
    titleDiv.innerText = currentChat.name;

    const dateDiv = document.createElement("div");
    dateDiv.style.textAlign = "right";
    dateDiv.style.fontSize = "12px";
    dateDiv.style.color = "#666";
    dateDiv.style.marginBottom = "10px";
    dateDiv.innerText = new Date().toLocaleDateString("ko-KR");

    const messagesContent = document
      .querySelector(".chat-content")
      .cloneNode(true);

    chatContent.appendChild(titleDiv);
    chatContent.appendChild(dateDiv);
    chatContent.appendChild(messagesContent);
    document.body.appendChild(chatContent);

    const canvas = await html2canvas(chatContent);
    const imgData = canvas.toDataURL("image/png");

    const doc = new jsPDF({
      orientation: "p",
      unit: "mm",
      format: "a4",
    });

    const imgWidth = 190;
    const imgHeight = (canvas.height * imgWidth) / canvas.width;
    doc.addImage(imgData, "PNG", 10, 10, imgWidth, imgHeight);

    document.body.removeChild(chatContent);

    doc.save(`chat_${currentChat.name}.pdf`);
  };

  return (
    <Box flex={1} p={6} display="flex" flexDirection="column">
      {currentChat && (
        <Box mb={4} borderBottom="1px solid" borderColor="gray.600" pb={4}>
          <HStack justify="space-between">
            {isEditing ? (
              <form onSubmit={handleNameSubmit} style={{ flex: 1 }}>
                <HStack>
                  <Input
                    value={chatName}
                    onChange={(e) => setChatName(e.target.value)}
                    size="md"
                    bg="gray.700"
                    color="white"
                    autoFocus
                  />
                  <Button type="submit" size="md" colorScheme="blue">
                    저장
                  </Button>
                  <Button
                    size="md"
                    variant="ghost"
                    onClick={() => setIsEditing(false)}
                  >
                    취소
                  </Button>
                </HStack>
              </form>
            ) : (
              <HStack justify="space-between" width="100%">
                <HStack>
                  <Text fontSize="xl" fontWeight="bold" color="white">
                    {currentChat.name}
                  </Text>
                  <Button
                    size="sm"
                    variant="ghost"
                    color="gray.300"
                    onClick={() => setIsEditing(true)}
                  >
                    수정
                  </Button>
                </HStack>
                <Button
                  size="sm"
                  colorScheme="blue"
                  onClick={handleDownloadPDF}
                  isDisabled={!currentChat.messages.length}
                >
                  PDF 다운로드
                </Button>
              </HStack>
            )}
          </HStack>
        </Box>
      )}

      <VStack
        flex={1}
        spacing={4}
        align="stretch"
        mb={4}
        overflowY="auto"
        className="chat-content"
      >
        {currentChat?.messages.map((msg, index) => (
          <Box
            key={index}
            bg={msg.type === "user" ? "blue.500" : "green.100"}
            p={4}
            borderRadius="md"
            alignSelf={msg.type === "user" ? "flex-end" : "flex-start"}
            maxW="70%"
          >
            <Text color={msg.type === "user" ? "white" : "green.700"}>
              {msg.content}
            </Text>
          </Box>
        ))}
      </VStack>

      <form onSubmit={handleSubmit}>
        <HStack spacing={4}>
          <Input
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="무엇을 도와드릴까요?"
            size="lg"
            bg="gray.700"
            border="none"
            color="white"
            _placeholder={{ color: "gray.400" }}
            _focus={{
              border: "1px solid",
              borderColor: "blue.500",
            }}
          />
          <Button type="submit" colorScheme="blue" size="lg">
            전송
          </Button>
        </HStack>
      </form>
    </Box>
  );
}

export default ChatArea;
