from pathlib import Path
from pydantic import BaseModel
from PIL import Image
from sentence_transformers import SentenceTransformer
from numpy import ndarray
from constants import DEBUG
from movies import MoviesList, Movie, load_movies
from inverted_index import DocID
from .semantic_search import cosine_similarity


class MultiModalSearchResult(BaseModel):
    doc_ID: DocID
    doc_title: str
    doc_descr: str
    similarity_score: float


class MultimodalSearch:
    MOVIE_FILE_PATH = Path("./data") / "movies.json"

    def __init__(self, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents: MoviesList = load_movies(MultimodalSearch.MOVIE_FILE_PATH.__str__())
        self.texts: list[str] = list(map(lambda x: f"{x.title}: {x.description}", self.documents.movies))
        self.text_embeddings: ndarray = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, image_path: str) -> ndarray:
        with open(image_path, "rb") as input_file:
            image_file = Image.open(input_file)
            embedding = self.model.encode([image_file], show_progress_bar=True)
        return embedding[0]

    @classmethod
    def search_with_image(cls, image_path: str, limit: int = 5) -> list[MultiModalSearchResult]:
        mmodal_search = MultimodalSearch()
        image_embedding = mmodal_search.embed_image(image_path)
        imtext_similarities = \
            list(
                map(
                    lambda x: cosine_similarity(x, image_embedding), 
                    mmodal_search.text_embeddings
                )
            )
        result: list[MultiModalSearchResult] = \
            list(
                map(
                    lambda x: MultiModalSearchResult(
                        doc_ID=x[1].id,
                        doc_title=x[1].title,
                        doc_descr=x[1].description,
                        similarity_score=imtext_similarities[x[0]]
                    ),
                    enumerate(mmodal_search.documents.movies)
                )
            )
        min_limit = min(limit, len(result))
        return [
            MultiModalSearchResult(
                doc_ID=x.doc_ID,
                doc_title=x.doc_title,
                doc_descr=x.doc_descr,
                similarity_score=x.similarity_score
            ) 
            for i, x in enumerate(sorted(result, key=lambda x: x.similarity_score, reverse=True)) 
            if i < min_limit
        ]



def verify_image_embedding(image_path: str) -> None:
    mmodal_search = MultimodalSearch()
    if DEBUG: print(f"Generating embedding for image at path '{image_path}'")
    image_embedding = mmodal_search.embed_image(image_path)
    print(f"Embedding shape: {image_embedding.shape[0]} dimensions")


def image_search_command(image_path: str) -> None:
    results = MultimodalSearch.search_with_image(image_path)
    for i, result in enumerate(results):
        print(f"{i+1}. {result.doc_title} (similarity: {result.similarity_score:.3f})")
        print(f"\t{result.doc_descr:.80}")