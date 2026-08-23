from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def generate_embedding(self, text: str):
        if len(text.strip()) == 0:
            raise ValueError("The 'text' parameter must be a non-empty string with content other than whitespace.")
        encoded_result = self.model.encode([text])
        return encoded_result[0]

def verify_model():
    semsearch = SemanticSearch()
    print(f"Model loaded: {semsearch.model}")
    print(f"Max sequence length: {semsearch.model.max_seq_length}")

def embed_text(text: str):
    semsearch = SemanticSearch()
    embedding = semsearch.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")