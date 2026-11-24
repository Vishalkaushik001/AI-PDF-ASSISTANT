
from sentence_transformers import SentenceTransformer
from src.vector_store import create_vector_store, search_vector_store

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample document chunks
chunks = [
    {
        "chunk_id": 1,
        "page": 1,
        "text": "Machine learning is a branch of artificial intelligence."
    },
    {
        "chunk_id": 2,
        "page": 2,
        "text": "Python is a popular programming language."
    },
    {
        "chunk_id": 3,
        "page": 3,
        "text": "Deep learning uses neural networks."
    },
    {
        "chunk_id": 4,
        "page": 4,
        "text": "Artificial intelligence enables machines to perform intelligent tasks."
    }
]

# Generate document embeddings
texts = [chunk["text"] for chunk in chunks]

embeddings = model.encode(
    texts,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print("Document embeddings generated!")

# Create FAISS index
index = create_vector_store(embeddings)

print("FAISS index created!")
print("Total vectors:", index.ntotal)

# User question
question = "What is machine learning?"

# Generate query embedding
query_embedding = model.encode(
    [question],
    convert_to_numpy=True,
    normalize_embeddings=True
)

# Search relevant chunks
results = search_vector_store(
    index,
    query_embedding[0],
    chunks,
    top_k=3
)

print("\nSearch Results:")

for result in results:

    print("\nChunk ID:", result["chunk"]["chunk_id"])

    print("Page:", result["chunk"]["page"])

    print("Text:", result["chunk"]["text"])

    print("Similarity Score:", result["score"])