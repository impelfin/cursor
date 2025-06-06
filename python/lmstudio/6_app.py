from lmstudio import BaseModel
import lmstudio as lms

# A class based schema for a book
class BookSchema(BaseModel):
    title: str
    author: str
    year: int

model = lms.llm()

result = model.respond("Tell me about The Hobbit", response_format=BookSchema)
book = result.parsed

print(book)