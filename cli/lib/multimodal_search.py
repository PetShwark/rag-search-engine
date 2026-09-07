from PIL import Image
from sentence_transformers import SentenceTransformer
from numpy import ndarray
from constants import DEBUG

class MultimodalSearch:
    def __init__(self, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)

    def embed_image(self, image_path: str) -> ndarray:
        with open(image_path, "rb") as input_file:
            image_file = Image.open(input_file)
            embedding = self.model.encode([image_file], show_progress_bar=True)
        return embedding[0]


def verify_image_embedding(image_path: str) -> None:
    mmodal_search = MultimodalSearch()
    if DEBUG: print(f"Generating embedding for image at path '{image_path}'")
    image_embedding = mmodal_search.embed_image(image_path)
    print(f"Embedding shape: {image_embedding.shape[0]} dimensions")