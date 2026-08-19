import string
from nltk.stem import PorterStemmer

def stem_tokens(input_tokens: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    #unique_strings = set()
    result: list[str] = []
    for input_token in input_tokens:            
        result.append(stem_token(input_token, stemmer))
    return result


def stem_token(input_token: str, stemmer: PorterStemmer) -> str:
    return stemmer.stem(depunctuate(input_token).lower().strip())


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
    Takes a string and returns tokens.  The tokens are all in lowercase,
    devoid of punctuation, and stemmed.
    """
    stemmer = PorterStemmer()
    return list(map(lambda x: stem_token(x, stemmer),filter(None, input.split())))


def tokenize_term(input: str) -> str:
    words_found = tokenize(input)
    if len(words_found) != 1:
        raise ValueError("Search term must be one token.")
    stemmer = PorterStemmer()
    return stem_token(depunctuate(words_found[0]).lower().strip(), stemmer)


def get_tokens_from_string(search_for: str, stop_words: list[str]) -> list[str]:
    # Split into words and get rid of blanks
    tokens_iter = filter(None, search_for.split()) # Iterator
    # Depuntuate, lowercase and strip whitespace from ends
    tokens_list = list(map(lambda x: depunctuate(x).lower().strip(),tokens_iter))
    tokens_list = filtered_tokens(stop_words, tokens_list)
    stemmed_tokens = stem_tokens(tokens_list)
    return stemmed_tokens