from sentence_transformers import SentenceTransformer
import chromadb


def load_retriever(chroma_path: str = "chroma_db"):
    """
    Load the ChromaDB collection and embedding model
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_collection("risk_documents")
    print("✅ Retriever loaded")
    return model, collection


def retrieve(query: str, model, collection, top_k: int = 5) -> dict:
    """
    Retrieve the most relevant chunks for a given query.
    Returns chunks, sources, and a confidence score.
    """
    # Embed the query
    query_embedding = model.encode([query]).tolist()

    # Search ChromaDB
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # Convert cosine distance to confidence score (0-1)
    # Lower distance = higher similarity
    confidence = 1 - min(distances)

    retrieved = []
    for chunk, meta, dist in zip(chunks, metadatas, distances):
        retrieved.append({
            "text": chunk,
            "source": meta["source"],
            "similarity": round(1 - dist, 4)
        })

    return {
        "query": query,
        "confidence": round(confidence, 4),
        "results": retrieved
    }


if __name__ == "__main__":
    model, collection = load_retriever()

    # Test queries
    test_queries = [
        "What is Barclays CET1 capital ratio?",
        "How does IFRS 9 define stage 2 credit impairment?",
        "What are the key credit risks facing Barclays?"
    ]

    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        result = retrieve(query, model, collection)
        print(f"Confidence: {result['confidence']}")
        for i, r in enumerate(result["results"][:2]):
            print(f"\n  Result {i+1} | Source: {r['source']} | Similarity: {r['similarity']}")
            print(f"  {r['text'][:200]}...")