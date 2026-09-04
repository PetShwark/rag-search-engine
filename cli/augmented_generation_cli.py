import argparse
from constants import RRF_K
from movies import load_movies, Movie
from lib.hybrid_search import HybridSearch
from lib.llm_stuff import rag_search_command, rag_summarize_command, rag_citations_command, rag_question_command


def rag_command(query: str) -> None:
    movie_list = load_movies(HybridSearch.MOVIES_JSON_FILE.__str__())
    hybrid_search = HybridSearch(movie_list.movies)
    search_results = hybrid_search.rrf_search(query, 60)
    llm_response_str = rag_search_command(query, search_results, hybrid_search.idx.docmap)
    print("Search Results:")
    for search_result in search_results:
        print(f"- {hybrid_search.idx.docmap[search_result['id']].title}")
    print("\nRAG Response:")
    print(f"{llm_response_str}")


def summarize_command(query: str) -> None:
    movie_list = load_movies(HybridSearch.MOVIES_JSON_FILE.__str__())
    hybrid_search = HybridSearch(movie_list.movies)
    search_results = hybrid_search.rrf_search(query, 60)
    llm_response_str = rag_summarize_command(query, search_results, hybrid_search.idx.docmap)
    print("Search Results:")
    for search_result in search_results:
        print(f"- {hybrid_search.idx.docmap[search_result['id']].title}")
    print("\nLLM Summary:")
    print(f"{llm_response_str}")



def llm_citations_command(query: str) -> None:
    movie_list = load_movies(HybridSearch.MOVIES_JSON_FILE.__str__())
    hybrid_search = HybridSearch(movie_list.movies)
    search_results = hybrid_search.rrf_search(query, 60)
    llm_response_str = rag_citations_command(query, search_results, hybrid_search.idx.docmap)
    print("Search Results:")
    for search_result in search_results:
        print(f"- {hybrid_search.idx.docmap[search_result['id']].title}")
    print("\nLLM Answer:")
    print(f"{llm_response_str}")



def llm_question_command(query: str, limit: int) -> None:
    movie_list = load_movies(HybridSearch.MOVIES_JSON_FILE.__str__())
    hybrid_search = HybridSearch(movie_list.movies)
    search_results = hybrid_search.rrf_search(query, 60, limit)
    llm_response_str = rag_question_command(query, search_results, hybrid_search.idx.docmap)
    print("Search Results:")
    for search_result in search_results:
        print(f"- {hybrid_search.idx.docmap[search_result['id']].title}")
    print("\nAnswer:")
    print(f"{llm_response_str}")



def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summary_parser = subparsers.add_parser(
        "summarize", help="Perform LLM summarization"
    )
    summary_parser.add_argument("query", type=str, help="Search query for RAG")

    citation_parser = subparsers.add_parser(
        "citations", help="Perform LLM summarization that includes citations in the response"
    )
    citation_parser.add_argument("query", type=str, help="Search query for RAG")

    question_parser = subparsers.add_parser(
        "question", help="Query results are passed to the LLM to answer questions about the documents in the query results"
    )
    question_parser.add_argument("query", type=str, help="Search query for RAG")
    question_parser.add_argument("--limit", type=int, default=5, help="Query result limit")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            rag_command(query)
        case "summarize":
            query = args.query
            summarize_command(query)
        case "citations":
            query = args.query
            llm_citations_command(query)
        case "question":
            query = args.query
            limit = args.limit
            llm_question_command(query, limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()