
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "I love machine learning.",
    "I enjoy studying artificial intelligence.",
    "The weather is very cold today."
]

embeddings = model.encode(
    sentences,
    normalize_embeddings=True
)

# Compare the first sentence with the others
similarity = cosine_similarity(
    [embeddings[0]],
    embeddings
)

print("Similarity scores:")

for i, sentence in enumerate(sentences):
    print(sentence)
    print(similarity[0][i])
    print()
