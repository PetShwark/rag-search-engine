import os
from typing import TYPE_CHECKING
from time import sleep
from dotenv import load_dotenv
from openai import OpenAI
if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam, ChatCompletion
from movies import Movie
if TYPE_CHECKING:
    from .hybrid_search import RRFScoreRecord
from .helper_funcs import rating_text_to_float



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


def llm_rewrite_query(query: str) -> str | None:
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
                "content":f"""Rewrite the user-provided movie search query below to be more specific and searchable.

                Consider:
                - Common movie knowledge (famous actors, popular films)
                - Genre conventions (horror = scary, animation = cartoon)
                - Keep the rewritten query concise (under 10 words)
                - It should be a Google-style search query, specific enough to yield relevant results
                - Don't use boolean logic

                Examples:
                - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

                If you cannot improve the query, output the original unchanged.
                Output only the rewritten query text, nothing else.

                User-provided movie search query: "{query}"
                """
            }
        ]
    completions = client.chat.completions.create(messages=messages, model="openrouter/free")
    if isinstance(completions, ChatCompletion):
        result = completions.choices[0].message.content 
    return result


def llm_expand_query(query: str) -> str | None:
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
                "content":f"""Expand the user-provided movie search query below with related terms.

                Add synonyms and related concepts that might appear in movie descriptions.
                Keep expansions relevant and focused.
                Output only the additional terms; they will be appended to the original query.

                Examples:
                - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
                - "action movie with bear" -> "action thriller bear chase fight adventure"
                - "comedy with bear" -> "comedy funny bear humor lighthearted"

                User-provided movie query: "{query}"
                """
            }
        ]
    completions = client.chat.completions.create(messages=messages, model="openrouter/free")
    if isinstance(completions, ChatCompletion):
        result = completions.choices[0].message.content 
    return result


def llm_rerank_query(query: str, movies: list[Movie]) -> list[tuple[Movie, float]]:
    """
    Take a user-supplied movie search query string and a list of Movie objects and use the
    OpenRouter/free model to see how well each movie matches the supplied query.  Return a
    list of the given movies each with a 0-10 match score supplied by the model.
    """
    result: list[tuple[Movie, float]] = []
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    for movie in movies:
        messages: list[ChatCompletionMessageParam] = \
            [
                {
                    "role":"user",
                    "content":f"""Rate how well this movie matches the search query.

                    Query: "{query}"
                    Movie: {movie.title} - {movie.description}

                    Consider:
                    - Direct relevance to query
                    - User intent (what they're looking for)
                    - Content appropriateness

                    Rate 0-10 (10 = perfect match).
                    Output ONLY the number in your response, no other text or explanation.

                    Score:"""
                }
            ]
        completions = client.chat.completions.create(messages=messages, model="openrouter/free")
        movie_score = 0.0
        if isinstance(completions, ChatCompletion):
            returned_content = completions.choices[0].message.content
            movie_score = float(returned_content) if isinstance(returned_content,str) else 0.0
        result.append((movie, movie_score))
        # Give the LLM a rest so we don't get throttled
        sleep(5) # seconds
    return result


def llm_rerank_rrf_search_results(query: str, search_results: list[RRFScoreRecord], docmap: dict[int, Movie]) -> list[tuple[float, RRFScoreRecord]]:
    """
    Use an LLM to give ratings on how well a given movie in the search results matches the given query.  The docmap is needed
    to lookup the title and description info for the llm to do the rating.  The serch results only has the movie's id.
    """
    accumulator: list[tuple[float, RRFScoreRecord]] = []
    # Accumulate llm ratings
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    for search_result in search_results:
        messages: list[ChatCompletionMessageParam] = \
            [
                {
                    "role":"user",
                    "content":f"""Rate how well this movie matches the search query.

                    Query: "{query}"
                    Movie: {docmap[search_result['id']].title} - {docmap[search_result['id']].description}

                    Consider:
                    - Direct relevance to query
                    - User intent (what they're looking for)
                    - Content appropriateness

                    Rate 0-10 (10 = perfect match).
                    Output ONLY the number in your response, no other text or explanation.

                    Score:"""
                }
            ]
        completions = client.chat.completions.create(messages=messages, model="openrouter/free")
        movie_score = 0.0
        if isinstance(completions, ChatCompletion):
            returned_content = completions.choices[0].message.content
            movie_score = rating_text_to_float(returned_content) if isinstance(returned_content,str) else 0.0
        accumulator.append((movie_score, search_result))
        # Give the LLM a rest so we don't get throttled
        sleep(5) # seconds
    return list(
        sorted(
            accumulator, 
            key=lambda x: x[0], 
            reverse=True
        )
    )