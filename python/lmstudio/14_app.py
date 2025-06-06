import lmstudio as lms

model = lms.llm()

tokens = model.tokenize("Hello, world!")

print(tokens)