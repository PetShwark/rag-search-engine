from sentence_transformers import SentenceTransformer
import numpy as np
from typing import TypedDict
from movies import Movie, load_movies
from constants import DEBUG, SEMANTIC_SCORE_PRECISION, SEMANTIC_SEARCH_DESCR_MAX_LEN
from pathlib import Path
from tqdm import tqdm
import json
import re


class SemanticSearch:
    EMBEDDINGS_SAVE_FILE = Path("./cache") / "movie_embeddings.npy"
    MOVIES_JSON_FILE = Path("./data") / "movies.json"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if DEBUG: print(f"Loading sentence transformer '{model_name}', please wait.")
        self.model = SentenceTransformer(model_name)
        if DEBUG: print("Done.")
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
        for index, (_, doc) in tqdm(enumerate(self.document_map.items()), desc="Scoring movies") if DEBUG else enumerate(self.document_map.items()):
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


class ChunkMetadata(TypedDict):
    movie_idx: int
    chunk_idx: int
    total_chunks: int


class ChunkScore(TypedDict):
    chunk_idx: int
    movie_idx: int
    score: float


class ChunkedSearchResult(TypedDict):
    id: int
    title: str
    document: str
    score: float
    metadata: ChunkMetadata | None


class MovieScore(TypedDict):
    chunk_idx: int
    score: float


class ChunkedSemanticSearch(SemanticSearch):
    EMBEDDINGS_SAVE_FILE = Path("./cache") / "movie_embeddings.npy"
    CHUNK_METADATA_JSON_FILE = Path("./cache") / "chunk_metadata.json"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[Movie]) -> np.ndarray:
        self.documents = documents
        all_chunks: list[str] = []
        all_metadata: list[ChunkMetadata] = []
        for doc in self.documents:
            self.document_map[doc.id] = doc
            if not doc.description:
                break
            doc_chunks = semantic_chunk(doc.description, max_chunk_size=4, overlap=1)
            num_chunks = len(doc_chunks)
            for chunk_idx, chunk in enumerate(doc_chunks):
                # if len(chunk) < 10:
                #     print(f"Tiny chunk: {chunk}, skipping")
                #     break
                all_chunks.append(chunk)
                all_metadata.append({
                    "movie_idx": doc.id,
                    "chunk_idx": chunk_idx,
                    "total_chunks": num_chunks
                })
        print(f"All chunks list size: {len(all_chunks)}")
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=DEBUG)
        self.chunk_metadata = all_metadata
        with open(ChunkedSemanticSearch.EMBEDDINGS_SAVE_FILE, "wb") as embeddings_file, \
            open(ChunkedSemanticSearch.CHUNK_METADATA_JSON_FILE, "w") as metadata_file:
            np.save(embeddings_file, self.chunk_embeddings)
            json.dump(self.chunk_metadata, metadata_file)
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[Movie]) -> np.ndarray:
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc.id] = doc
        if ChunkedSemanticSearch.EMBEDDINGS_SAVE_FILE.exists() and \
            ChunkedSemanticSearch.CHUNK_METADATA_JSON_FILE.exists():
            with open(ChunkedSemanticSearch.EMBEDDINGS_SAVE_FILE, "rb") as embeddings_file, \
                open(ChunkedSemanticSearch.CHUNK_METADATA_JSON_FILE, "r") as metadata_file:
                self.chunk_embeddings = np.load(embeddings_file)
                self.chunk_metadata = json.load(metadata_file)
            return self.chunk_embeddings
        else:
            return self.build_chunk_embeddings(documents)

    def search_chunked(self, query: str, limit: int = 10) -> list[ChunkedSearchResult]:
        """
        Use chunk embeddings saved in the object to compare and save cosine similarity scores.
        From the saved scores return the limit-highest score results with the following attributes:
            - movie_id
            - movie title
            - movie description
            - score of chunk of movie with highest score
            - metadata for the chunk that had the highest score for this movie
        """
        result: list[ChunkedSearchResult] = []
        # THere's a problem if these fields don't exist
        if not isinstance(self.chunk_embeddings, np.ndarray) or self.chunk_metadata == None:
            return result
        query_embedding = self.generate_embedding(query)
        # With the query embedding calculate cosine similarity scores for every chunk embedding 
        # NOTE: The order of embeddings is the same as the order of the chunks as they were processed.
        # NOTE: SO, index from enumerate is the index of the chunk from the originally semantically chunked list of all chunks.
        chunk_scores: list[ChunkScore] = []
        for index, chunk_embedding in enumerate(self.chunk_embeddings):
            chunk_scores.append(ChunkScore(
                chunk_idx = self.chunk_metadata[index]["chunk_idx"],
                movie_idx = self.chunk_metadata[index]["movie_idx"],
                score = cosine_similarity(query_embedding, chunk_embedding)
            ))
        # Go through list of scores for all the chunks and gather up the max scores for each movie, saving
        # the highest score found and the chunk's index within the movie desciption.
        movie_scores: dict[int, MovieScore] = {}
        for chunk_score in chunk_scores:
            if movie_scores.get(chunk_score["movie_idx"], None):
                if chunk_score["score"] > movie_scores[chunk_score["movie_idx"]]["score"]:
                    movie_scores[chunk_score["movie_idx"]] = \
                        MovieScore(
                            chunk_idx=chunk_score["chunk_idx"],
                            score=chunk_score["score"]
                        )
            else: 
                movie_scores[chunk_score["movie_idx"]] = \
                    MovieScore(
                        chunk_idx=chunk_score["chunk_idx"],
                        score=chunk_score["score"]
                    )
        new_limit = min(limit, len(movie_scores))
        # Get limit number of scores sorted in descending order of score
        limited_movie_scores = {
            movie_id: score 
            for i, (movie_id, score) in enumerate(sorted(movie_scores.items(), key=lambda x: x[1]["score"], reverse=True)) 
            if i < new_limit
        }
        
        # Assemble the result structures
        for movie_id, score in limited_movie_scores.items():
            descr = self.document_map[movie_id].description if self.document_map.get(movie_id, None) else ""
            descr_len = min(len(descr), SEMANTIC_SEARCH_DESCR_MAX_LEN)
            result.append(ChunkedSearchResult(
                id = movie_id,
                score = round(score["score"], SEMANTIC_SCORE_PRECISION),
                title = self.document_map[movie_id].title if self.document_map.get(movie_id, None) else "",
                document = descr[:descr_len],
                metadata = self.chunk_metadata[score["chunk_idx"]] if self.chunk_metadata != None else None
            ))
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


def semantic_chunk(text: str, max_chunk_size: int, overlap: int) -> list[str]:
    """
    Chunks on sentence boundaries, then merges sentences into chunks of size <= max_chunk_size, with overlap between chunks.
    """
    if len(text.strip()) == 0:
        return []
    # Strip first, then split on punctuation (if any), then filter out empty strings
    sentences = list(filter(None, map(lambda x: x.strip(), re.split(r"(?<=[.!?])\s+", text.strip()))))
    if len(sentences) == 1 and not sentences[0].endswith(('.','?','!')):
        return [ text.strip() ]
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
    return chunks


def semantic_chunk_command(text: str, max_chunk_size: int, overlap: int) -> None:
    chunks = semantic_chunk(text, max_chunk_size, overlap)
    print(f"Semantically chunking {len(text.strip())} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i+1}. {chunk}")


def embed_chunks_command() -> None:
    documents = load_movies(SemanticSearch.MOVIES_JSON_FILE.__str__())
    chunked_semsearch = ChunkedSemanticSearch()
    embeddings = chunked_semsearch.load_or_create_chunk_embeddings(documents.movies)
    print(f"Generated {len(embeddings)} chunked embeddings.")


def search_chunked_command(query: str, limit: int) -> None:
    documents = load_movies(SemanticSearch.MOVIES_JSON_FILE.__str__())
    chunked_semsearch = ChunkedSemanticSearch()
    _ = chunked_semsearch.load_or_create_chunk_embeddings(documents.movies)
    results = chunked_semsearch.search_chunked(query, limit)
    print("Search results:")
    for index, result in enumerate(results):
        print(f"\n{index+1}. {result['title']} (score: {result['score']:.4f})")
        print(f"\t{result['document']}")