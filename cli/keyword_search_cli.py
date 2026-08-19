import argparse
from movies import load_movies, Movie
from tokenizer import stop_words, filtered_tokens, tokenize, token_found, stem_tokens, tokenize_term
from inverted_index import InvertedIndex, DocID

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


def search_command(search_for: str, max_results: int = 5) -> None:
    i = InvertedIndex()
    try:
        i.load()
    except:
        print(f"Unable to load indices from disk.  Perhaps build them?")
        return
    search_tokens = stem_tokens(filtered_tokens(InvertedIndex.STOP_WORDS_LIST, tokenize(search_for)))
    found_movie_ids: list[DocID] = []
    for search_token in search_tokens:
        found_movie_ids.extend(i.get_documents(search_token))
    result_count = 0
    print(f"Searching for: {search_for}")
    print(f"Search results:")
    for found_id in found_movie_ids:
        if result_count >= max_results:
            print(f"Max number ({max_results}) results reached.")
            break
        else:
            print(f"{found_id} {i.docmap[found_id].title}")
            result_count += 1


def build_command() -> None:
    i = InvertedIndex()
    i.build(json_file_name="./data/movies.json")
    i.save()


def tf_command(movie_id: int, term: str) -> None:
    i = InvertedIndex()
    try:
        i.load()
    except:
        print(f"Unable to load indices from disk.  Perhaps build them?")
        return
    try:
        validate_term = tokenize_term(term) # raises an error if term is not a single token
    except ValueError:
        print(f"Search term must be a single token.")
        return
    tf = i.get_tf(movie_id, validate_term)
    print(f"Term frequency of '{validate_term}' in '{i.docmap[movie_id].title}' ({movie_id}) is {tf}")


def idf_command(term: str) -> None:
    pass



def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Import movie data and build indices")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a given Movie ID")
    tf_parser.add_argument("movie_id", type=int, help="Movie ID number")
    tf_parser.add_argument("term", type=str, help="Search term to get frequency of")

    idf_parser = subparsers.add_parser("idf", help="Calculate the TF-IDF for a given term")
    idf_parser.add_argument("term", type=str, help="The term for which to calculate the TF-IDF")

    args = parser.parse_args()

    match args.command:
        case "search":
            # print the search query here
            search_command(args.query)
        case "build":
            build_command()
        case "tf":
            tf_command(args.movie_id, args.term)
        case "idf":
            idf_command(args.term)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()