from lmstudio import BaseModel
import lmstudio as lms

BookSchema = {
    "type": "object",
    "properties": {
        "title": { "type": "string" },
        "author": { "type": "string" },
        "year": { "type": "integer" },
    },
    "required": ["title", "author", "year"],
}
model = lms.llm()

prediction_stream = model.respond_stream("Tell me about The Hobbit", response_format=BookSchema)

for fragment in prediction_stream:
    print(fragment.content, end="", flush=True)
