import argparse
import math
from tokenizer import get_tokens_from_string, validate_single_token_from_string
from inverted_index import InvertedIndex, DocID
from constants import BM25_K1, BM25_B

def load_index(i: InvertedIndex) -> InvertedIndex | None:
    """
    Takes an InvertedIndex instance and runs the .load() method on it.
    If no error, returns the same instance, else returns None.
    """
    try:
        i.load()
    except:
        print(f"Unable to load indices from disk.  Perhaps build them?")
        return
    return i

def search_command(i: InvertedIndex, search_for: str, max_results: int = 5) -> None:
    """
    Command function for searching for movies given search terms in a string.
    Uses the term index in the given InvertedIndex for matching.
    """
    if not (loaded_index := load_index(i)):
        return
    search_tokens = get_tokens_from_string(search_for, InvertedIndex.STOP_WORDS_LIST)
    found_movie_ids: set[DocID] = set()
    for search_token in search_tokens:
        found_movie_ids.update(loaded_index.get_documents(search_token))
    result_count = 0
    print(f"Searching for: {search_for}")
    print(f"Search results:")
    for found_id in sorted(list(found_movie_ids)):
        if result_count >= max_results:
            print(f"Max number ({max_results}) results reached.")
            break
        else:
            print(f"({found_id}) {loaded_index.docmap[found_id].title}")
            result_count += 1


def build_command(i: InvertedIndex) -> None:
    """
    Command function to build and save the InvertedIndex object.
    1) Pulls in the JSON file of movie data.
    2) Processes movie data structure to create the docmap, index, 
        term_frequencies, and doc_lengths data structures.
    3) Saves the data from the above structures into Python pickle 
        files in the './cache' folder (assumes running tool in the
        project root folder)
    """
    print("Building index from './data/movies.json'...")
    i.build(json_file_name="./data/movies.json")
    print("Saving pickles to './cache'...")
    i.save()
    print("Done.")


def tf_command(i: InvertedIndex, movie_id: int, term: str) -> None:
    """
    Command function for pulling up the TF for a given movie and term.
    The 'term' string must contain ONE token or a ValueError is raised.
    """
    if not (loaded_index := load_index(i)):
        return    
    term_token = validate_single_token_from_string(term, InvertedIndex.STOP_WORDS_LIST) #RAISES
    tf = loaded_index.get_tf(movie_id, term_token)
    print(f"Term frequency of '{term}' ('{term_token}', tokenized) in '{loaded_index.docmap[movie_id].title}' ({movie_id}) is {tf}")


def idf_command(i: InvertedIndex, term: str) -> None:
    """
    Command function for calculating the IDF for a given term.
    The 'term' string must contain ONE token or a ValueError is raised.
    """
    if not (loaded_index := load_index(i)):
        return
    term_token = validate_single_token_from_string(term, InvertedIndex.STOP_WORDS_LIST) #RAISES
    movie_count = len(loaded_index.docmap)
    term_movies_count = len(loaded_index.index[term_token]) if loaded_index.index.get(term_token) else 0
    idf = math.log(float(movie_count + 1) / float(term_movies_count + 1))
    print(f"Movie count: {movie_count}")
    print(f"Movies with term: {term_movies_count}")
    print(f"Inverse document frequency of '{term}': {idf:.2f}")


def tfidf_command(i: InvertedIndex, movie_id: int, term: str) -> None:
    """
    Command function for calculating the TF-IDF for a given movie and term.
    The 'term' string must contain ONE token or a ValueError is raised.
    """
    if not (loaded_index := load_index(i)):
        return
    term_token = validate_single_token_from_string(term, InvertedIndex.STOP_WORDS_LIST) #RAISES
    movie_count = len(loaded_index.docmap)
    term_movies_count = len(loaded_index.index[term_token]) if loaded_index.index.get(term_token) else 0
    idf = math.log(float(movie_count + 1) / float(term_movies_count + 1))
    tf = loaded_index.get_tf(movie_id, term_token)
    tfidf = float(tf) * idf
    print(f"Movies with term: {term_movies_count}")
    print(f"IDF: {idf}")
    print(f"TF: {tf}")
    print(f"TF-IDF score of '{term}' ('{term_token}', tokenized) in document '{movie_id}': {tfidf:.2f}")


def bm25_idf_command(i: InvertedIndex, term: str) -> float:
    """
    Command function for calculating the BM25 IDF for a given term.
    The 'term' string must contain ONE token or a ValueError is raised.
    """
    if not (loaded_index := load_index(i)):
        return 0.0
    term_token = validate_single_token_from_string(term, InvertedIndex.STOP_WORDS_LIST) #RAISES
    return loaded_index.get_bm25_idf(term_token)


def bm25_tf_command(i: InvertedIndex, movie_id: int, term: str, k1: float, b: float) -> float:
    """
    Command function for calculating the BM25 TF for a given movie and term.
    The 'term' string must contain ONE token or a ValueError is raised.
    The tuning parameters k1 and b [0.0 <-> 1.0] may be provided.
    """
    if not (loaded_index := load_index(i)):
        return 0.0
    term_token = validate_single_token_from_string(term, InvertedIndex.STOP_WORDS_LIST)
    return loaded_index.get_bm25_tf(movie_id, term_token, k1, b)


def doc_lengths_command(i: InvertedIndex, movie_id: int) -> None:
    """
    Command function to pull up the entire list of doc_lengths in the InvertedIndex, or
    (if a non-zero movie_id is given) the doc_length for a given movie (None if movie_id
    doesn't exist).
    """
    if not (loaded_index := load_index(i)):
        return
    if movie_id > 0:
        print(f"Doc length for Movie #{movie_id} is {loaded_index.doc_lengths.get(movie_id)}")
    else:
        print("Doc length dictionary:")
        print(loaded_index.doc_lengths)


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

    bm25idf_parser = subparsers.add_parser("bm25idf", help="Calculate the BM25-IDF for a given term")
    bm25idf_parser.add_argument("term", type=str, help="The term for which to calculate the BM25-IDF")

    bm25tf_parser = subparsers.add_parser("bm25tf", help="Calculate the BM25-TF for the given movie (via ID) and term")
    bm25tf_parser.add_argument("movie_id", type=int, help="Movie ID number")
    bm25tf_parser.add_argument("term", type=str, help="The term for which to calculate the BM25-TF")
    bm25tf_parser.add_argument("k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25tf_parser.add_argument("b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter")

    doc_lengths_parser = subparsers.add_parser("doclengths", help="Dump the doc_lengths dictionary")
    doc_lengths_parser.add_argument("movie_id", type=int, nargs="?", default=0, help="Movie ID number (optional)")

    args = parser.parse_args()

    i = InvertedIndex()

    match args.command:
        case "search":
            # print the search query here
            search_command(i, args.query)
        case "build":
            build_command(i)
        case "tf":
            tf_command(i, args.movie_id, args.term)
        case "idf":
            idf_command(i, args.term)
        case "tfidf":
            tfidf_command(i, args.movie_id, args.term)
        case "bm25idf":
            bm25_idf = bm25_idf_command(i, args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25_idf:.2f}")
        case "bm25tf":
            print(f"K1 is {args.k1}")
            print(f"B is {args.b}")
            bm25tf = bm25_tf_command(i, args.movie_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.movie_id}': {bm25tf:.2f}")
        case "doclengths":
            doc_lengths_command(i, args.movie_id)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()