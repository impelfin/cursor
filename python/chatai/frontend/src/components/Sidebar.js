import React from "react";
import { Box, Button, VStack, Text, HStack } from "@chakra-ui/react";

function Sidebar({
  chats,
  currentChat,
  onChatSelect,
  onNewChat,
  onDeleteChat,
}) {
  const formatChatDate = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleString("ko-KR", {
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    });
  };

  return (
    <Box w="260px" bg="gray.900" p={4}>
      <VStack spacing={4} align="stretch">
        <Button
          onClick={onNewChat}
          colorScheme="blue"
          variant="solid"
          size="lg"
          w="100%"
          bg="blue.500"
          color="white"
          _hover={{ bg: "blue.600" }}
          boxShadow="md"
        >
          + New Chat
        </Button>

        <VStack spacing={2} align="stretch">
          {chats.map((chat) => (
            <Box
              key={chat.id}
              p={3}
              bg={currentChat?.id === chat.id ? "gray.700" : "transparent"}
              borderRadius="md"
              cursor="pointer"
              _hover={{ bg: "gray.700" }}
            >
              <HStack justify="space-between">
                <Box onClick={() => onChatSelect(chat)}>
                  <Text color="white" fontWeight="bold" mb={1}>
                    {chat.name}
                  </Text>
                  <Text color="gray.400" fontSize="sm">
                    {formatChatDate(chat.id)}
                  </Text>
                </Box>
                <Button
                  size="sm"
                  colorScheme="red"
                  variant="ghost"
                  onClick={(e) => {
                    e.stopPropagation();
                    onDeleteChat(chat.id);
                  }}
                >
                  삭제
                </Button>
              </HStack>
            </Box>
          ))}
        </VStack>
      </VStack>
    </Box>
  );
}

export default Sidebar;
