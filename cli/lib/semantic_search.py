from sentence_transformers import SentenceTransformer
import numpy as np
from movies import Movie, load_movies
from pathlib import Path
from tqdm import tqdm
import re


class SemanticSearch:
    EMBEDDINGS_SAVE_FILE = Path("./cache") / "movie_embeddings.npy"
    MOVIES_JSON_FILE = Path("./data") / "movies.json"

    def __init__(self):
        print(f"Loading sentence transformer 'all-MiniLM-L6-v2', please wait.")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Done.")
        self.embeddings: np.ndarray | None = None
        self.documents: list[Movie] = []
        self.document_map: dict[int, Movie] = {}

    def generate_embedding(self, text: str):
        text = text.strip()
        if len(text) == 0:
            raise ValueError(
                "The 'text' parameter must be a non-empty string with content other than whitespace.")
        encoded_result = self.model.encode([text])
        return encoded_result[0]

    def build_embeddings(self, documents: list[Movie]):
        self.documents = documents
        strings_to_generate = []
        for doc in self.documents:
            self.document_map[doc.id] = doc
            strings_to_generate.append(f"{doc.title} {doc.description}")
        self.embeddings = self.model.encode(
            strings_to_generate, show_progress_bar=True)
        with open(SemanticSearch.EMBEDDINGS_SAVE_FILE, "wb") as file:
            np.save(file, self.embeddings)
        return self.embeddings

    def load_or_build_embeddings(self, documents: list[Movie]):
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc.id] = doc
        if SemanticSearch.EMBEDDINGS_SAVE_FILE.exists():
            with open(SemanticSearch.EMBEDDINGS_SAVE_FILE, "rb") as file:
                self.embeddings = np.load(file)
                if isinstance(self.embeddings, np.ndarray) and len(self.embeddings) == len(self.documents):
                    return self.embeddings
                else:
                    print(
                        f"Loaded embedding list differs in size from the documents list! REGENERATING!")
                    return self.build_embeddings(documents)
        else:
            print(f"Embeddings save file does not exist. GENERATING embeddings.")
            return self.build_embeddings(documents)

    def search(self, query: str, limit: int) -> list[dict[str, float | str]]:
        result: list[dict[str, float | str]] = []
        if not isinstance(self.embeddings, np.ndarray):
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        similarity_scores: list[tuple[float, Movie]] = []
        # Keys should be in same order as added to dictionary
        for index, (_, doc) in tqdm(enumerate(self.document_map.items()), desc="Scoring movies"):
            similarity_scores.append(
                (cosine_similarity(query_embedding, self.embeddings[index]), doc))
        sorted_scores = sorted(
            similarity_scores, key=lambda x: x[0], reverse=True)
        # Just in case a crazy limit value is supplied
        new_limit = min(limit, len(self.document_map))
        result = [
            {
                "score": score_tuple[0],
                "title": score_tuple[1].title,
                "description": score_tuple[1].description
            }
            for i, score_tuple in enumerate(sorted_scores) if i < new_limit
        ]
        return result


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


def embed_query_text(query: str) -> None:
    semsearch = SemanticSearch()
    embedding = semsearch.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def search_command(query: str, limit: int) -> None:
    semsearch = SemanticSearch()
    documents = load_movies(SemanticSearch.MOVIES_JSON_FILE.__str__())
    _ = semsearch.load_or_build_embeddings(documents.movies)
    search_results = semsearch.search(query, limit)
    print("Search results:")
    for index, search_result in enumerate(search_results):
        title = search_result.get("title", "MISSING TITLE")
        score = search_result.get("score", "MISSING SCORE")
        description = search_result.get("description", "MISSING DESCRIPTION")
        print(f"{index+1}. {title} (score: {score})\n\t{description:.80}")


def chunk_command(text: str, chunk_size: int, overlap: int) -> None:
    words = text.strip().split()
    total_words = len(words)
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")
    chunk_begin = 0
    chunk_end = chunk_size
    chunks: list[str] = []
    while chunk_end < len(words):
        # print(f"begin: {chunk_begin}, end: {chunk_end}")
        chunks.append(" ".join(words[chunk_begin:chunk_end]))
        chunk_begin += chunk_size - overlap
        chunk_end += chunk_size - overlap
    # print(f"begin: {chunk_begin}, end: {chunk_end}")
    if chunk_begin < len(words):
        chunks.append(" ".join(words[chunk_begin:]))
    print(f"Chunking {len(text.strip())} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i+1}. {chunk}")


def semantic_chunk_command(text: str, max_chunk_size: int, overlap: int) -> None:
    """
    Chunks on sentence boundaries, then merges sentences into chunks of size <= max_chunk_size, with overlap between chunks.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if overlap < 0 or overlap >= max_chunk_size:
        raise ValueError("overlap must be >= 0 and < max_chunk_size")
    chunk_begin = 0
    chunk_end = max_chunk_size
    chunks: list[str] = []
    while chunk_end < len(sentences):
        # print(f"begin: {chunk_begin}, end: {chunk_end}")
        chunks.append(" ".join(sentences[chunk_begin:chunk_end]))
        chunk_begin += max_chunk_size - overlap
        chunk_end += max_chunk_size - overlap
    # print(f"begin: {chunk_begin}, end: {chunk_end}")
    if chunk_begin < len(sentences):
        chunks.append(" ".join(sentences[chunk_begin:]))
    print(f"Semantically chunking {len(text.strip())} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i+1}. {chunk}")
