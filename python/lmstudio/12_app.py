import lmstudio as lms
from create_file_tool import create_file

model = lms.llm()
model.act(
    "Please create a file named output.txt with your understanding of the meaning of life.",
    [create_file],
)

print("File created.")