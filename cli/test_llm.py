import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.environ.get("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

messages = [
    {
        "role": "user",
        "content": "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum.",
    }
]
completion = client.chat.completions.create(messages=messages, model="openrouter/free")

print(f"Response: {completion.choices[0].message.content}")
print(f"Prompt tokens: {completion.usage.prompt_tokens}")
print(f"Response tokens: {completion.usage.completion_tokens}")