import argparse
from constants import RRF_K
from movies import load_movies, Movie
from lib.hybrid_search import HybridSearch
from lib.llm_stuff import rag_search_command


def rag_command(query: str) -> None:
    movie_list = load_movies(HybridSearch.MOVIES_JSON_FILE.__str__())
    hybrid_search = HybridSearch(movie_list.movies)
    search_results = hybrid_search.rrf_search(query, RRF_K, 5)
    llm_response_str = rag_search_command(query, search_results, hybrid_search.idx.docmap)
    print("Search Results:")
    for search_result in search_results:
        print(f"- {hybrid_search.idx.docmap[search_result['id']].title}")
    print("\nRAG Response:")
    print(f"{llm_response_str}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            rag_command(query)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()