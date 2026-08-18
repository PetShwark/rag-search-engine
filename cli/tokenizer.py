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
            stemmed_search_for_token = stemmer.stem(search_for_token)
            stemmed_search_in_token = stemmer.stem(search_in_token)
            if stemmed_search_for_token in stemmed_search_in_token:
                return True
    return False


def stem_tokens(input_tokens: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    unique_strings = set()
    for input_token in input_tokens:
        unique_strings.add(stemmer.stem(input_token))
    return list(unique_strings)


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
