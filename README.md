# Credit Risk RAG Intelligence System

An AI-powered question-answering system for financial risk documents,
built as a prototype for production deployment in a banking environment.

## Overview
- Ingests regulatory and financial PDFs (Basel III, IFRS 9, Barclays Annual Report)
- Chunks and embeds documents into a ChromaDB vector store
- Retrieves relevant context via semantic search
- Generates cited, auditable answers via AWS Bedrock (Claude 3 Sonnet)
- Includes confidence-based guardrails and full query logging
  for Model Risk Policy compliance

## Tech Stack
- Python, LangChain, AWS Bedrock, ChromaDB, Streamlit, RAGAs

## Architecture
[you'll add a diagram here later]

## Setup
[you'll fill this in as you build]
```

---

### What Your Repo Should Look Like After This Step
```
barclays-risk-rag/
├── data/
│   └── raw_pdfs/        ← local only, gitignored
├── src/                 ← empty for now, ready for code
├── app/                 ← empty for now
├── logs/                ← empty for now
├── .env                 ← local only, gitignored
├── .gitignore           ✅ committed
├── requirements.txt     ✅ committed
└── README.md            ✅ committed
