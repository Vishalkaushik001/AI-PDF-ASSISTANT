
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "Machine learning is a branch of artificial intelligence.",
    "AI allows computers to learn from data.",
    "I like eating pizza."
]

embeddings = model.encode(
    sentences,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print("Embedding shape:", embeddings.shape)

print("\nFirst sentence embedding:")
print(embeddings[0])

print("\nNumber of dimensions:")
print(len(embeddings[0]))
