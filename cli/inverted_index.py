from pydantic import BaseModel
from typing import List, Dict, Set, ClassVar, Counter
from pickle import dump, load
from pathlib import Path
from tokenizer import tokenize, stop_words, filtered_tokens, stem_tokens
from movies import load_movies, Movie

type Token = str
type DocID = int

class InvertedIndex(BaseModel):
    STOP_WORDS_LIST: ClassVar[List[str]] = stop_words("./data/stopwords.txt")

    index: Dict[Token, Set[DocID]] = {}
    docmap: Dict[DocID, Movie] = {}
    term_frequencies: Dict[DocID, Counter] = {}

    def get_tf(self, doc_id: DocID, term: str) -> int:
        result = 0
        if self.term_frequencies.get(doc_id):
            result = self.term_frequencies[doc_id][term]
        return result

    def __add_document(self, doc_id: DocID, doc: Movie) -> None:
        doc_tokens = tokenize(f"{doc.title} {doc.description}")
        doc_tokens = filtered_tokens(InvertedIndex.STOP_WORDS_LIST, doc_tokens)
        doc_tokens = stem_tokens(doc_tokens)
        token_set = set(doc_tokens)
        self.docmap[doc_id] = doc
        for doc_token in doc_tokens:
            if not self.term_frequencies.get(doc_id):
                self.term_frequencies[doc_id] = Counter()
            self.term_frequencies[doc_id][doc_token] += 1
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
        movies_list = load_movies(json_file_name)
        for movie in movies_list.movies:
            doc_id = movie.id
            self.__add_document(doc_id, movie)

    def save(self) -> None:
        # Make the cache folder. Don't freak if exists
        cache_folder = Path("./cache")
        cache_folder.mkdir(parents=True, exist_ok=True)
        index_file_path = cache_folder / "index.pkl"
        docmap_file_path = cache_folder / "docmap.pkl"
        term_frequencies_file_path = cache_folder / "term_frequencies.pkl"
        with open(index_file_path, "wb") as index_file, \
            open(docmap_file_path, "wb") as docmap_file, \
            open(term_frequencies_file_path, "wb") as term_frequencies_file:
            dump(self.index, index_file)
            dump(self.docmap, docmap_file)
            dump(self.term_frequencies, term_frequencies_file)

    def load(self) -> None:
        cache_folder = Path("./cache")
        index_file_path = cache_folder / "index.pkl"
        docmap_file_path = cache_folder / "docmap.pkl"
        term_frequencies_file_path = cache_folder / "term_frequencies.pkl"
        with open(index_file_path, "rb") as index_file, \
            open(docmap_file_path, "rb") as docmap_file, \
            open(term_frequencies_file_path, "rb") as term_frequencies_file:
            self.index = load(index_file)
            self.docmap = load(docmap_file)
            self.term_frequencies = load(term_frequencies_file)
