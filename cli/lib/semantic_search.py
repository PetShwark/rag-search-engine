from sentence_transformers import SentenceTransformer
import numpy as np
from movies import Movie, load_movies
from pathlib import Path

class SemanticSearch:
    EMBEDDINGS_SAVE_FILE = Path("./cache") / "movie_embeddings.npy"
    MOVIES_JSON_FILE = Path("./data") / "movies.json"

    def __init__(self):
        print(f"Loading sentence transformer 'all-MiniLM-L6-v2', please wait.")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Done.")
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text: str):
        if len(text.strip()) == 0:
            raise ValueError("The 'text' parameter must be a non-empty string with content other than whitespace.")
        encoded_result = self.model.encode([text])
        return encoded_result[0]

    def build_embeddings(self, documents: list[Movie]):
        self.documents = documents
        strings_to_generate = []
        for doc in self.documents:
            self.document_map[doc.id] = doc
            strings_to_generate.append(f"{doc.title} {doc.description}")
        self.embeddings = self.model.encode(strings_to_generate, show_progress_bar=True)
        with open(SemanticSearch.EMBEDDINGS_SAVE_FILE, "wb") as file:
            np.save(file, self.embeddings)
        return self.embeddings

    def load_or_build_embeddings(self, documents: list[Movie]):
        self.documents = documents
        strings_to_generate = []
        for doc in self.documents:
            self.document_map[doc.id] = doc
        if SemanticSearch.EMBEDDINGS_SAVE_FILE.exists():
            with open(SemanticSearch.EMBEDDINGS_SAVE_FILE, "rb") as file:
                self.embeddings = np.load(file)
                if len(self.embeddings) == len(self.documents):
                    return self.embeddings
                else:
                    print(f"Loaded embedding list differs in size from the documents list! REGENERATING!")
                    return self.build_embeddings(documents)
        else:
            print(f"Embeddings save file does not exist. GENERATING embeddings.")
            return self.build_embeddings(documents)


def verify_model() -> None:
    semsearch = SemanticSearch()
    print(f"Model loaded: {semsearch.model}")
    print(f"Max sequence length: {semsearch.model.max_seq_length}")

def embed_text(text: str) -> None:
    semsearch = SemanticSearch()
    embedding = semsearch.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings() -> None:
    semsearch = SemanticSearch()
    documents = load_movies(SemanticSearch.MOVIES_JSON_FILE.__str__())
    embeddings = semsearch.load_or_build_embeddings(documents.movies)
    print(f"Number of docs:   {len(documents.movies)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )