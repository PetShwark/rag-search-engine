import string
from nltk.stem import PorterStemmer


def stem_words(input_words: list[str]) -> list[str]:
    """
    Takes a list of strings (assumed to be depunctuated, lower-cased and stripped)
    and returns a new list with the strings run through a PortStemmer().
    """
    stemmer = PorterStemmer()
    result: list[str] = []
    for input_token in input_words:            
        result.append(stemmer.stem(input_token))
    return result


def filtered_words(filter_out: list[str], from_list: list[str]) -> list[str]:
    """
    Remove members of filter_out from from_list.

    The words in each list are assumed to be depunctuated, lower-cased and stripped.
    """
    result: list[str] = []
    for word in from_list:
        remove_this_word = False
        for bad_word in filter_out:
            if bad_word == word:
                remove_this_word = True
                break
        if not remove_this_word:
            result.append(word)
    return result


def depunctuate(input: str) -> str:
    """
    Removes punctuation characters (string.punctuation) from the input string.

    Needs 'string' module.
    """
    punctranslator = str.maketrans("", "", string.punctuation)
    return input.translate(punctranslator)


# def tokenize(input: str) -> list[str]:
#     """
#     Takes a string and returns tokens.  The tokens are all in lowercase,
#     devoid of punctuation, and stemmed.
#     """
#     stemmer = PorterStemmer()
#     return list(map(lambda x: stemmer.stem(depunctuate(x).lower().strip()),filter(None, input.split())))


# def tokenize_term(input: str) -> str:
#     words_found = tokenize(input)
#     if len(words_found) != 1:
#         raise ValueError("Search term must be one token.")
#     stemmer = PorterStemmer()
#     return stemmer.stem(depunctuate(words_found[0]).lower().strip())


def get_tokens_from_string(input: str, stop_words: list[str]) -> list[str]:
    """
    Takes an input string and tokenizes it by:
    1) Splitting on whitespace,
    2) Removing empties,
    3) Removing punctuation marks ('string' module's string.punctuation)
    4) Converting to lower-case,
    5) Removing leading and trailing whitespace,
    6) Filtering out "stop words" (list given as argument),
    7) Stemming each word with a PortStemmer.
    """
    filtered_word_list = \
        filtered_words(
            stop_words, 
            list(
                map(
                    lambda x: depunctuate(x).lower().strip(),
                    filter(None, input.split())
                )
            )
        )
    stemmed_tokens = stem_words(filtered_word_list)
    return stemmed_tokens