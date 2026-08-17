import argparse
import json
import string
from nltk.stem import PorterStemmer

def token_found(search_for: list[str], search_in: list[str]) -> bool:
    """
    Looks for the strings in the search_for list in the search_in list of strings.
    It is assumed that the strings have been tokenized (lowercase-d and punctuation
    removed).
    """
    stemmer = PorterStemmer()
    for search_in_token in search_in:
        for search_for_token in search_for:
            stemmed_for_token = stemmer.stem(search_for_token)
            stemmed_in_token = stemmer.stem(search_in_token)
            if stemmed_for_token in stemmed_in_token:
                return True
    return False


def stop_words(filename: str) -> list[str]:
    """
    Gets the stop words from the named file and processes them into tokens.  It returns the list of tokens.
    """
    result: list[str] = []
    with open(filename, "r") as file:
        lines = file.read().splitlines()
    for word in lines:
        result.append(depunctuate(word).lower().strip())
    return result


def filtered_tokens(filter_out: list[str], from_list: list[str]) -> list[str]:
    result: list[str] = []
    for from_token in from_list:
        remove_this = False
        for token_to_remove in filter_out:
            if token_to_remove == from_token:
                remove_this = True
                break
        if not remove_this:
            result.append(from_token)
    return result


def depunctuate(input: str) -> str:
    punctranslator = str.maketrans("", "", string.punctuation)
    return input.translate(punctranslator)


def tokenize(input: str) -> list[str]:
    """
    Takes a string and returns tokens.  The tokens are all in lowercase and
    devoid of punctuation.
    """
    result: list[str] = []
    words = input.split()
    for word in words:
        if word:
            result.append(depunctuate(word).lower().strip())
    return result


def search_movies_from_json_file(json_file: str, search_for: str, max_results: int = 5) -> list[str]:
    result: list[str] = []
    with open(json_file, "r") as file:
        try:
            movie_data = json.load(file)
        except json.JSONDecodeError, UnicodeDecodeError:
            print(f"Error decoding JSON file.")
    result_count = 0
    stop_word_tokens = stop_words("./data/stopwords.txt")
    search_for_tokens = filtered_tokens(filter_out=stop_word_tokens, from_list=tokenize(search_for))
    if movie_data.get("movies"):
        for movie in movie_data["movies"]:
            if result_count >= max_results:
                break
            if movie.get("title"):
                movie_title_tokens = filtered_tokens(filter_out=stop_word_tokens, from_list=tokenize(movie["title"]))
                if token_found(search_for_tokens, movie_title_tokens):
                    result.append(movie["title"])
                    result_count += 1
    return result
            

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            # print the search query here
            print(f"Searching for: {args.query}")
            search_results = search_movies_from_json_file("./data/movies.json", args.query)
            for index, title in enumerate(search_results):
                print(f"{index+1}. {title}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()