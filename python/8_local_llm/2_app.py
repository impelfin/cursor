from openai import OpenAI

client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', 
)

response = client.chat.completions.create(
    model="llama3.2:latest",        
    messages=[
        {
            "role": "system",
            "content": [
            {
                "type": "text",
                "text": "You are a python expert."
            }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text":  "Code a Python function to generate a Fibonacci sequence."
                }
            ]
        }
    ],
)
result = response.choices[0].message.content
print(result)