from pydantic import BaseModel
from typing import List, Dict, Set
from pickle import dump
from pathlib import Path
from tokenizer import tokenize
from movies import load_movies

type Token = str
type DocID = int
type Document = str

class InvertedIndex(BaseModel):
    index: Dict[Token, Set[DocID]]
    docmap: Dict[DocID, Document]

    def __add_document(self, doc_id: DocID, doc: Document) -> None:
        doc_tokens = tokenize(doc)
        for doc_token in doc_tokens:
            self.docmap[doc_id] = doc
            if not self.index.get(doc_token):
                self.index[doc_token] = set()
            self.index[doc_token].add(doc_id)

    def get_documents(self, term: Token) -> List[DocID]:
        result: List[DocID] = []
        if self.index.get(term):
            result = list(self.index[term])
            result.sort()
        return result

    def build(self, json_file_name: str) -> None:
        movies_list = load_movies(json_file_name)
        for movie in movies_list.movies:
            doc = f"{movie.title} - {movie.description}"
            doc_id = movie.id
            self.__add_document(doc_id, doc)

    def save(self) -> None:
        # Make the cache folder. Don't freak if exists
        cache_folder = Path("./cache")
        cache_folder.mkdir(parents=True, exist_ok=True)
        index_file_path = cache_folder / "index.pkl"
        docmap_file_path = cache_folder / "docmap.pkl"
        with open(index_file_path, "wb") as index_file, open(docmap_file_path, "wb") as docmap_file:
            dump(self.index, index_file)
            dump(self.docmap, docmap_file)