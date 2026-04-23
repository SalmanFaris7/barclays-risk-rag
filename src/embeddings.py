from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import json
from pathlib import Path


def load_chunks(chunks_path: str) -> list:
    """
    Load chunks produced by ingestion.py
    """
    with open(chunks_path, "r") as f:
        chunks = json.load(f)
    print(f"✅ Loaded {len(chunks)} chunks")
    return chunks


def embed_and_store(chunks: list, chroma_path: str = "chroma_db"):
    """
    Embed chunks using sentence-transformers and store in ChromaDB
    """
    # Load embedding model
    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Initialise ChromaDB
    client = chromadb.PersistentClient(path=chroma_path)
    
    # Create or connect to collection
    collection = client.get_or_create_collection(
        name="risk_documents",
        metadata={"hnsw:space": "cosine"}
    )

    # Prepare data
    texts = [chunk["text"] for chunk in chunks]
    ids = [chunk["chunk_id"] for chunk in chunks]
    metadatas = [{"source": chunk["source"]} for chunk in chunks]
    # Removing the noise
    chunks = [c for c in chunks if len(c["text"].strip()) > 50]
    print(f"✅ After filtering noise: {len(chunks)} chunks")

    # Embed in batches
    print(f"Embedding {len(texts)} chunks...")
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        batch_metadatas = metadatas[i:i + batch_size]

        embeddings = model.encode(batch_texts).tolist()

        collection.add(
            documents=batch_texts,
            embeddings=embeddings,
            ids=batch_ids,
            metadatas=batch_metadatas
        )

        print(f"  Stored batch {i // batch_size + 1} / {(len(texts) // batch_size) + 1}")

    print(f"✅ All chunks embedded and stored in ChromaDB at '{chroma_path}'")
    return collection


def verify_store(chroma_path: str = "chroma_db"):
    """
    Quick sanity check — query the store with a test question
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_collection("risk_documents")

    test_query = "What is the CET1 capital ratio?"
    embedding = model.encode([test_query]).tolist()

    results = collection.query(
        query_embeddings=embedding,
        n_results=3
    )

    print("\n🔍 Test query:", test_query)
    print("\nTop 3 results:")
    for i, (doc, meta) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0]
    )):
        print(f"\n--- Result {i+1} ---")
        print(f"Source: {meta['source']}")
        print(f"Text: {doc[:200]}...")


if __name__ == "__main__":
    # Step 1 — load chunks from ingestion
    chunks = load_chunks("data/all_chunks.json")

    # Step 2 — embed and store
    embed_and_store(chunks)

    # Step 3 — verify it worked
    verify_store()