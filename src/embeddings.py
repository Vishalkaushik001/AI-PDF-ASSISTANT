
from sentence_transformers import SentenceTransformer


# Load the pretrained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def generate_embeddings(chunks):

    # Extract text from each chunk
    texts = [chunk["text"] for chunk in chunks]

    if not texts:
        return []

    # Convert text into numerical vectors
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return embeddings
