import os
from typing import TYPE_CHECKING
from time import sleep
from json import loads
from dotenv import load_dotenv
from openai import OpenAI
if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam, ChatCompletion
    from inverted_index import DocID
    from movies import Movie
    from .hybrid_search import RRFScoreRecord
from .helper_funcs import rating_text_to_float


def get_llm_client() -> OpenAI:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable not set")
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    return client


def llm_spell_check(query: str) -> str | None:
    result = None
    client = get_llm_client()
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
    client = get_llm_client()
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
    client = get_llm_client()
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


def llm_rerank_rrf_search_results(query: str, search_results: list[RRFScoreRecord], docmap: dict[int, Movie]) -> list[tuple[float, RRFScoreRecord]]:
    """
    Use an LLM to give ratings on how well a given movie in the search results matches the given query.  The docmap is needed
    to lookup the title and description info for the llm to do the rating.  The serch results only has the movie's id.
    """
    client = get_llm_client()
    accumulator: list[tuple[float, RRFScoreRecord]] = []
    # Accumulate llm ratings
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


def llm_batch_rerank_rrf_search_results(query: str, search_results: list[RRFScoreRecord], docmap: dict[int, Movie]) -> list[DocID]:
    client = get_llm_client()
    # Build the list of movies for the prompt
    movie_list_text = ""
    for search_result in search_results:
        movie_list_text += f"""
        
        Movie ID: {search_result['id']},
        Movie Title: {docmap[search_result['id']].title},
        Movie Description: {docmap[search_result['id']].description}
        """
    messages: list[ChatCompletionMessageParam] = \
        [
            {
                "role":"user",
                "content":f"""Rank the movies listed below by relevance to the following search query.

                Search Query: "{query}"

                Movies:
                {movie_list_text}

                Return the Movie IDs in order of relevance, best match first.

                Your response must be a raw JSON array of integers.
                Do not wrap the JSON in Markdown. Do not use a ```json code block.
                Do not include any explanatory text.

                For example:
                [75, 12, 34, 2, 1]

                Ranking:"""
            }
        ]
    completions = client.chat.completions.create(messages=messages, model="openrouter/free")
    if movie_id_json := completions.choices[0].message.content:
        ranked_movie_ids: list[int] = loads(movie_id_json)
    return ranked_movie_ids


def llm_evaluate_search_results(query: str, search_results: list[RRFScoreRecord], docmap: dict[int, Movie]) -> list[int]:
    """
    Use an LLM prompt to evaluate the RRF movie search results given.  The docmap is provided in order to 
    get movie title and description information for the prompt.  The prompt specifically requests that 
    the LLM return a JSON list of integers [0,3] as the ratings of the movies in the order of the search 
    results.
    """
    client = get_llm_client()
    formatted_results: list[str] = []
    for index, result in enumerate(search_results):
        result_str = f"""{index+1}. {docmap[result['id']].title}
        \tRRF score: {result['data']['rrf_score']}
        \tBM25 rank: {result['data']['bm25_rank']}, Semantic rank: {result['data']['sem_rank']}
        \t{docmap[result['id']].description}
        """
        formatted_results.append(result_str)
    messages: list[ChatCompletionMessageParam] = \
        [
            {
                "role":"user",
                "content":f"""Rate how relevant each result is to this query on a 0-3 scale:

                Query: "{query}"

                Results:
                {chr(10).join(formatted_results)}

                Scale:
                - 3: Highly relevant
                - 2: Relevant
                - 1: Marginally relevant
                - 0: Not relevant

                Do NOT give any numbers other than 0, 1, 2, or 3.

                Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

                [2, 0, 3, 2, 0, 1]"""
            }
        ]
    completions = client.chat.completions.create(messages=messages, model="openrouter/free")
    json_list_str = completions.choices[0].message.content
    json_list = []
    if json_list_str:
        json_list = loads(json_list_str)
    return json_list



def rag_search_command(query: str, search_results: list[RRFScoreRecord], docmap: dict[DocID, Movie]) -> str:
    client = get_llm_client()
    # Assemble docs list for prompt
    docs = ""
    for search_result in search_results:
        docs += f"- {docmap[search_result['id']].title}\n"
    messages: list[ChatCompletionMessageParam] = \
        [
            {
                "role":"user",
                "content":f"""You are a RAG agent for Webflyx, a movie streaming service.
                Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
                Provide a comprehensive answer that addresses the user's query.
                You may make suggestions for documents that are not listed in the provided list of documents.

                Query: {query}

                Documents:
                {docs}

                Answer:"""
            }
        ]
    completions = client.chat.completions.create(messages=messages, model="openrouter/free")
    result = completions.choices[0].message.content
    return result if result else ""