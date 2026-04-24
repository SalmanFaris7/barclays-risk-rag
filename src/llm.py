from openai import OpenAI
import boto3
import json
import os
from dotenv import load_dotenv
from datetime import datetime, timezone
from retriever import load_retriever, retrieve

load_dotenv()

CONFIDENCE_THRESHOLD = 0.25

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


def call_bedrock(prompt: str) -> str:
    """PRIMARY — AWS Bedrock Claude 3 Sonnet"""
    client = boto3.client(
        service_name="bedrock-runtime",
        region_name=os.getenv("AWS_REGION", "eu-west-2")
    )
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}]
    })
    response = client.invoke_model(
        modelId="anthropic.claude-3-sonnet-20240229-v1:0",
        body=body
    )
    return json.loads(response["body"].read())["content"][0]["text"]


def call_deepseek(prompt: str) -> str:
    """FALLBACK 1 — DeepSeek"""
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com"
    )
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1024,
        temperature=0.0
    )
    return response.choices[0].message.content


def call_openai(prompt: str) -> str:
    """FALLBACK 2 — OpenAI GPT-4o"""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1024,
        temperature=0.0
    )
    return response.choices[0].message.content


def call_llm_with_fallback(prompt: str) -> tuple:
    """
    Try each LLM in order. Returns (answer, provider_used).
    Bedrock → DeepSeek → OpenAI
    """
    providers = [
        ("AWS Bedrock", call_bedrock),
        ("DeepSeek",    call_deepseek),
        ("OpenAI",      call_openai),
    ]

    for provider_name, call_fn in providers:
        try:
            print(f"  Trying {provider_name}...")
            answer = call_fn(prompt)
            print(f"  ✅ Success with {provider_name}")
            return answer, provider_name
        except Exception as e:
            print(f"  ⚠️ {provider_name} failed: {e}. Trying next fallback...")

    return (
        "❌ All LLM providers failed. Please check your API keys and connectivity.",
        "None"
    )


def log_query(query: str, answer: str, sources: list, confidence: float, provider: str):
    """Audit log for Model Risk compliance"""
    os.makedirs("logs", exist_ok=True)
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "confidence": confidence,
        "sources": sources,
        "provider_used": provider,
        "answer": answer
    }
    with open("logs/query_log.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")


def answer_query(query: str, model, collection) -> dict:
    """Full RAG pipeline: retrieve → guardrail → generate → log"""

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
        log_query(query, answer, sources, confidence, provider="Guardrail")
        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "provider_used": "Guardrail",
            "guardrail_triggered": True
        }

    # Step 3 — Generate with fallback chain
    prompt = build_prompt(query, chunks)
    answer, provider = call_llm_with_fallback(prompt)

    # Step 4 — Log
    log_query(query, answer, sources, confidence, provider)

    return {
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
        "provider_used": provider,
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
        print(f"Provider used: {result['provider_used']}")
        print(f"Guardrail triggered: {result['guardrail_triggered']}")
        print(f"Sources: {result['sources']}")
        print(f"\nAnswer:\n{result['answer']}")