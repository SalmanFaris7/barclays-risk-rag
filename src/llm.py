from openai import OpenAI
import os
import json
from dotenv import load_dotenv
from datetime import datetime, timezone
from retriever import load_retriever, retrieve

load_dotenv()

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.25

# DeepSeek client — uses OpenAI-compatible SDK
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

SYSTEM_PROMPT = """You are a senior credit risk analyst at a major UK bank.
You answer questions strictly based on the provided document excerpts.
Always cite which document your answer comes from.
If the context is insufficient, say so clearly — do not speculate.
Keep answers concise, precise and professional."""


def build_prompt(query: str, retrieved_chunks: list) -> str:
    context = ""
    for i, chunk in enumerate(retrieved_chunks):
        context += f"\n[Source {i+1}: {chunk['source']}]\n{chunk['text']}\n"

    return f"""Use the following document excerpts to answer the question.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""


def call_deepseek(prompt: str) -> str:
    """
    Call DeepSeek API using OpenAI-compatible SDK
    """
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1024,
        temperature=0.0  # Keep deterministic for risk use case
    )
    return response.choices[0].message.content


def log_query(query: str, answer: str, sources: list, confidence: float):
    """
    Audit log every query for Model Risk compliance
    """
    os.makedirs("logs", exist_ok=True)
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "confidence": confidence,
        "sources": sources,
        "answer": answer
    }
    with open("logs/query_log.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")


def answer_query(query: str, model, collection) -> dict:
    """
    Full RAG pipeline: retrieve → guardrail → generate → log
    """
    # Step 1 — Retrieve
    retrieval = retrieve(query, model, collection)
    confidence = retrieval["confidence"]
    chunks = retrieval["results"]
    sources = list(set([c["source"] for c in chunks]))

    # Step 2 — Guardrail
    if confidence < CONFIDENCE_THRESHOLD:
        answer = (
            "⚠️ Insufficient information in the available documents "
            "to answer this question reliably. Please consult source documents directly."
        )
        log_query(query, answer, sources, confidence)
        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "guardrail_triggered": True
        }

    # Step 3 — Generate
    prompt = build_prompt(query, chunks)
    answer = call_deepseek(prompt)

    # Step 4 — Log
    log_query(query, answer, sources, confidence)

    return {
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
        "guardrail_triggered": False
    }


if __name__ == "__main__":
    model, collection = load_retriever()

    test_queries = [
        "What is Barclays CET1 capital ratio?",
        "How does IFRS 9 define stage 2 credit impairment?",
        "What are the key credit risks facing Barclays?"
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        result = answer_query(query, model, collection)
        print(f"Confidence: {result['confidence']}")
        print(f"Guardrail triggered: {result['guardrail_triggered']}")
        print(f"Sources: {result['sources']}")
        print(f"\nAnswer:\n{result['answer']}")