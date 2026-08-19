import argparse
from tokenizer import get_tokens_from_string
from inverted_index import InvertedIndex, DocID


def search_command(search_for: str, max_results: int = 5) -> None:
    i = InvertedIndex()
    try:
        i.load()
    except:
        print(f"Unable to load indices from disk.  Perhaps build them?")
        return
    search_tokens = get_tokens_from_string(search_for, InvertedIndex.STOP_WORDS_LIST)
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
    term_tokens = get_tokens_from_string(term, InvertedIndex.STOP_WORDS_LIST) # raises an error if term is not a single word
    if len(term_tokens) != 1:
        raise ValueError(f"Term must be one token.  '{term}' is not.")
    term_token = term_tokens[0]
    tf = i.get_tf(movie_id, term_token)
    print(f"Term frequency of '{term}' ('{term_token}', tokenized) in '{i.docmap[movie_id].title}' ({movie_id}) is {tf}")


def idf_command(term: str) -> None:
    import math
    i = InvertedIndex()
    try:
        i.load()
    except:
        print(f"Unable to load indices from disk.  Perhaps build them?")
        return
    term_tokens = get_tokens_from_string(term, InvertedIndex.STOP_WORDS_LIST) # raises an error if term is not a single word
    if len(term_tokens) != 1:
        raise ValueError(f"Term must be one token.  '{term}' is not.")
    term_token = term_tokens[0]
    movie_count = len(i.docmap)
    term_movies_count = 0
    if i.index.get(term_token):
        term_movies_count = len(i.index[term_token])
    idf = math.log(float(movie_count + 1) / float(term_movies_count + 1))
    print(f"Inverse document frequency of '{term}': {idf:.2f}")


def tfidf_command(movie_id: int, term: str) -> None:
    import math
    i = InvertedIndex()
    try:
        i.load()
    except:
        print(f"Unable to load indices from disk.  Perhaps build them?")
        return
    term_tokens = get_tokens_from_string(term, InvertedIndex.STOP_WORDS_LIST) # raises an error if term is not a single word
    if len(term_tokens) != 1:
        raise ValueError(f"Term must be one token.  '{term}' is not.")
    term_token = term_tokens[0]
    movie_count = len(i.docmap)
    term_movies_count = 0
    if i.index.get(term_token):
        term_movies_count = len(i.index[term_token])
    idf = math.log(float(movie_count + 1) / float(term_movies_count + 1))
    term_token = term_tokens[0]
    tf = i.get_tf(movie_id, term_token)
    tfidf = float(tf) * idf
    print(f"TF-IDF score of '{term}' ('{term_token}', tokenized) in document '{movie_id}': {tfidf:.2f}")



def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Import movie data and build indices")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a given Movie ID")
    tf_parser.add_argument("movie_id", type=int, help="Movie ID number")
    tf_parser.add_argument("term", type=str, help="Search term to get frequency of")

    idf_parser = subparsers.add_parser("idf", help="Calculate the IDF for a given term")
    idf_parser.add_argument("term", type=str, help="The term for which to calculate the IDF")

    tfidf_parser = subparsers.add_parser("tfidf", help="Calculate the TF-IDF value for the given movie (via ID) and term")
    tfidf_parser.add_argument("movie_id", type=int, help="Movie ID number")
    tfidf_parser.add_argument("term", type=str, help="The term for which to calculate the TF-IDF")

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
        case "tfidf":
            tfidf_command(args.movie_id, args.term)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()