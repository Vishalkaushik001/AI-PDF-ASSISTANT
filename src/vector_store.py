
import faiss
import numpy as np

def create_vector_store(embeddings):

    # Convert embeddings into NumPy float32 format
    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    # Check whether embeddings exist
    if embeddings.ndim != 2 or len(embeddings) == 0:
        raise ValueError("Embeddings must be a non-empty 2D array.")

    # Get the number of dimensions
    dimension = embeddings.shape[1]

    # Create FAISS index using Inner Product
    index = faiss.IndexFlatIP(dimension)

    # Add embeddings to FAISS
    index.add(embeddings)

    return index

def search_vector_store(index, query_embedding, chunks, top_k=3):

    if index.ntotal == 0:
        return []

    # Convert query embedding into float32 format
    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    )

    # Ensure query has the correct shape
    query_embedding = query_embedding.reshape(1, -1)

    if query_embedding.shape[1] != index.d:
        raise ValueError("Query embedding dimension does not match index.")

    # Avoid requesting more results than available
    top_k = min(top_k, index.ntotal)

    # Search for nearest vectors
    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    # Retrieve original chunk information
    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        results.append({
            "chunk": chunks[int(idx)],
            "score": float(score)
        })

    return results