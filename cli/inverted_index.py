from pydantic import BaseModel
from typing import List, Dict, Set, ClassVar, Counter
from pickle import dump, load
from pathlib import Path
from tqdm import tqdm
from tokenizer import get_tokens_from_string, depunctuate
from movies import load_movies, Movie
from constants import DEBUG, BM25_K1, BM25_B

type Token = str
type DocID = int


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


class InvertedIndex(BaseModel):
    STOP_WORDS_LIST: ClassVar[List[str]] = stop_words("./data/stopwords.txt")
    CACHE_DIR: ClassVar[Path] = Path("./cache")
    INDEX_PICKLE_PATH: ClassVar[Path] = CACHE_DIR / "index.pkl"
    DOCMAP_PICKLE_PATH: ClassVar[Path] = CACHE_DIR / "docmap.pkl"
    TERM_FREQ_PICKLE_PATH: ClassVar[Path] = CACHE_DIR / "term_frequencies.pkl"
    DOC_LENGTHS_PICKLE_PATH: ClassVar[Path] = CACHE_DIR / "doc_lengths.pkl"

    index: Dict[Token, Set[DocID]] = {}
    docmap: Dict[DocID, Movie] = {}
    term_frequencies: Dict[DocID, Counter] = {}
    doc_lengths: Dict[DocID, int] = {}

    def __get_avg_doc_length(self) -> float:
        num_docs = len(self.docmap)
        if num_docs == 0:
            return 0.0
        sum_doc_lengths = sum(self.doc_lengths.values())
        return float(sum_doc_lengths) / float(num_docs)

    def get_bm25_tf(self, doc_id: DocID, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        this_doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + (b * this_doc_length / avg_doc_length)
        tf = self.get_tf(doc_id, term)
        # print(f"Doc length: {this_doc_length}")
        # print(f"Avg Doc length: {avg_doc_length}")
        # print(f"Length norm: {length_norm}")
        # print(f"TF: {tf}")
        return ((float(tf) * (k1 + 1)) / (float(tf) + (k1 * length_norm)))

    def get_bm25_idf(self, term: str) -> float:
        from math import log
        num_docs = float(len(self.docmap))
        doc_freq = float(len(self.index[term])) if self.index.get(term) else 0.0
        bm25_idf = log(((num_docs - doc_freq + 0.5) / (doc_freq + 0.5)) + 1.0)
        return bm25_idf

    def get_tf(self, doc_id: DocID, term: str) -> int:
        result = 0
        if self.term_frequencies.get(doc_id):
            result = self.term_frequencies[doc_id][term]
        return result

    def __add_document(self, doc_id: DocID, doc: Movie) -> None:
        movie_text = f"{doc.title} {doc.description}"
        token_list = get_tokens_from_string(movie_text, InvertedIndex.STOP_WORDS_LIST)
        token_set = set(token_list)
        self.docmap[doc_id] = doc
        self.doc_lengths[doc_id] = len(token_list)
        for token in token_list: # All tokens from movie_text, includes repeats
            if not self.term_frequencies.get(doc_id):
                self.term_frequencies[doc_id] = Counter()
            self.term_frequencies[doc_id][token] += 1
        for token in token_set:
            if not self.index.get(token):
                self.index[token] = set()
            self.index[token].add(doc_id)

    def get_documents(self, term: Token) -> List[DocID]:
        result: List[DocID] = []
        if self.index.get(term):
            result = list(self.index[term])
            result.sort()
        return result

    def build(self, json_file_name: str) -> None:
        if DEBUG: print(f"Reading {json_file_name}...")
        movies_list = load_movies(json_file_name)
        for movie in tqdm(movies_list.movies):
            doc_id = movie.id
            self.__add_document(doc_id, movie)

    def save(self) -> None:
        # Make the cache folder. Don't freak if exists
        InvertedIndex.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(InvertedIndex.INDEX_PICKLE_PATH, "wb") as index_file, \
            open(InvertedIndex.DOCMAP_PICKLE_PATH, "wb") as docmap_file, \
            open(InvertedIndex.TERM_FREQ_PICKLE_PATH, "wb") as term_frequencies_file, \
            open(InvertedIndex.DOC_LENGTHS_PICKLE_PATH, "wb") as doc_lengths_file:
            if DEBUG: print("Saving index...")
            dump(self.index, index_file)
            if DEBUG: print("Saving docmap...")
            dump(self.docmap, docmap_file)
            if DEBUG: print("Saving term frequencies...")
            dump(self.term_frequencies, term_frequencies_file)
            if DEBUG: print("Saving document lengths...")
            dump(self.doc_lengths, doc_lengths_file)
            if DEBUG: print("Done.")

    def load(self) -> None:
        with open(InvertedIndex.INDEX_PICKLE_PATH, "rb") as index_file, \
            open(InvertedIndex.DOCMAP_PICKLE_PATH, "rb") as docmap_file, \
            open(InvertedIndex.TERM_FREQ_PICKLE_PATH, "rb") as term_frequencies_file, \
            open(InvertedIndex.DOC_LENGTHS_PICKLE_PATH, "rb") as doc_lengths_file:
            if DEBUG: print("Loading index...")
            self.index = load(index_file)
            if DEBUG: print("Loading docmap...")
            self.docmap = load(docmap_file)
            if DEBUG: print("Loading term frequencies...")
            self.term_frequencies = load(term_frequencies_file)
            if DEBUG: print("Loading document lengths...")
            self.doc_lengths = load(doc_lengths_file)
            if DEBUG: print("Done.\n")

    def bm25(self, doc_id: DocID, term: str) -> float:
        return self.get_bm25_tf(doc_id, term, BM25_K1, BM25_B) * self.get_bm25_idf(term)

    def bm25search(self, query: str, limit: int) -> dict[DocID, float]:
        query_tokens = get_tokens_from_string(query, InvertedIndex.STOP_WORDS_LIST)
        if DEBUG: print(f"Tokens from processed query string: {" ".join(query_tokens)}")
        accumulator: dict[DocID, float] = {}
        for query_token in query_tokens:
            movies_for_token = self.index.get(query_token, set())
            for movie_id in movies_for_token:
                accumulator[movie_id] = accumulator.get(movie_id, 0.0) + self.bm25(movie_id, query_token)
        sorted_accumulator = dict(sorted(accumulator.items(), key=lambda item: item[1], reverse=True))
        new_limit = min(limit, len(sorted_accumulator))
        return {movie_id: score for index, (movie_id, score) in enumerate(sorted_accumulator.items()) if index < new_limit}

