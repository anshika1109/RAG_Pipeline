"""
Step 3: Embeddings + Pinecone Storage

This module:
1. Converts text chunks into vector embeddings using sentence-transformers
2. Stores those vectors in Pinecone for later retrieval

Key Concepts:
- Embeddings are numerical representations of text (arrays of floats)
- Similar text = similar embeddings = close together in vector space
- Pinecone stores these vectors and lets us search by similarity
"""
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
)


def get_embedding_model():
    """
    Load the sentence-transformers model.
    'all-MiniLM-L6-v2' is small (80MB) and fast — good for learning.
    It produces 384-dimensional vectors.
    """
    print(f"🧠 Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"   Model loaded! Output dimension: {model.get_sentence_embedding_dimension()}")
    return model


def embed_chunks(model, chunks):
    """
    Convert document chunks into vector embeddings.

    Each chunk's text becomes a list of 384 floats that captures
    its semantic meaning.
    """
    texts = [chunk.page_content for chunk in chunks]
    print(f"🔢 Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)
    print(f"   Done! Each embedding has {len(embeddings[0])} dimensions")
    return embeddings


def store_in_pinecone(chunks, embeddings):
    """
    Store embeddings in Pinecone.

    Each vector is stored with:
    - id: a unique identifier
    - values: the embedding (list of floats)
    - metadata: the original text + source file (for retrieval later)
    """
    # Connect to Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    print(f"\n📌 Storing {len(chunks)} vectors in Pinecone index '{PINECONE_INDEX_NAME}'...")

    # Prepare vectors for upsert
    vectors = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vectors.append({
            "id": f"chunk-{i}",
            "values": embedding.tolist(),
            "metadata": {
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", "unknown"),
            },
        })

    # Upsert in batches of 100 (Pinecone best practice)
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch)
        print(f"   Upserted batch {i // batch_size + 1} ({len(batch)} vectors)")

    # Check index stats
    stats = index.describe_index_stats()
    print(f"\n✅ Pinecone index stats:")
    print(f"   Total vectors: {stats['total_vector_count']}")


if __name__ == "__main__":
    # Quick test: embed a single sentence
    model = get_embedding_model()
    test_embedding = model.encode(["Hello, this is a test"])
    print(f"\nTest embedding shape: {test_embedding.shape}")
    print(f"First 5 values: {test_embedding[0][:5]}")
