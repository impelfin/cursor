import lmstudio as lms

def does_chat_fit_in_context(model: lms.LLM, chat: lms.Chat) -> bool:
    formatted = model.apply_prompt_template(chat)
    token_count = len(model.tokenize(formatted))
    context_length = model.get_context_length()
    print(f"Token count: {token_count}, Context length: {context_length}")
    return token_count < context_length

model = lms.llm()

chat = lms.Chat.from_history({
    "messages": [
        { "role": "user", "content": "What is the meaning of life." },
        { "role": "assistant", "content": "The meaning of life is..." },
    ]
})

print("Fits in context:", does_chat_fit_in_context(model, chat))
