import argparse
from movies import load_movies
from tokenizer import stop_words, filtered_tokens, tokenize, token_found
from inverted_index import InvertedIndex

def search_movies_from_json_file(json_file: str, search_for: str, max_results: int = 5) -> list[str]:
    result: list[str] = []
    movie_data = load_movies(json_file)
    result_count = 0
    stop_word_tokens = stop_words("./data/stopwords.txt")
    search_for_tokens = filtered_tokens(filter_out=stop_word_tokens, from_list=tokenize(search_for))
    for movie in movie_data.movies:
        if result_count >= max_results:
            break
        if movie.title:
            movie_title_tokens = filtered_tokens(filter_out=stop_word_tokens, from_list=tokenize(movie.title))
            if token_found(search_for_tokens, movie_title_tokens):
                result.append(movie.title)
                result_count += 1
    return result


def build_command() -> None:
    i = InvertedIndex(docmap={}, index={})
    i.build(json_file_name="./data/movies.json")
    i.save()
    test_token = "merida"
    docs = i.get_documents(test_token)
    print(f"First document for token '{test_token}' = {docs[0]}")
            

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Import movie data and build indices")

    args = parser.parse_args()

    match args.command:
        case "search":
            # print the search query here
            print(f"Searching for: {args.query}")
            search_results = search_movies_from_json_file("./data/movies.json", args.query)
            for index, title in enumerate(search_results):
                print(f"{index+1}. {title}")
        case "build":
            build_command()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()