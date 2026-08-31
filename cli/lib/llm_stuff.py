import os
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletion


def llm_spell_check(query: str) -> str | None:
    result = None
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    messages: list[ChatCompletionMessageParam] = \
        [
            {
                "role":"user",
                "content":f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User movie search query: "{query}"
"""
            }
        ]
    completions = client.chat.completions.create(messages=messages, model="openrouter/free")
    if isinstance(completions, ChatCompletion):
        result = completions.choices[0].message.content 
    return result