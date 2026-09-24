# Module 3 — Support Assistant

This module provides a grounded GenAI support assistant built using LangGraph, ChromaDB, and FastAPI.

## RAG Pipeline Architecture
- **Ingestion & Embedding**: Loads 8 policy documents (`doc_01.txt`–`doc_08.txt`) from `/docs`, chunks each file, embeds them using `sentence-transformers/all-MiniLM-L6-v2`, and persists vectors in `ChromaDB`.
- **LangGraph Routing (3 Nodes)**:
  1. `classify_intent`: Keyword heuristic classifies queries as `policy_question` or `general_question`.
  2. `retrieve_and_answer`: Routes policy queries to fetch top-3 chunks from ChromaDB and formats a grounded response (`MOCK_LLM=1` default).
  3. `direct_answer`: Routes general queries to return a canned refusal without retrieval.
- **FastAPI API**: Exposes a `POST /ask` endpoint returning structured Pydantic JSON (`answer`, `sources`, `confidence`).

## Example API Call Transcripts (`MOCK_LLM=1` Default)

### Example 1: Policy Question (Triggers Retrieval)
**Request (`POST /ask`)**:
```json
{
  "query": "What is the return policy for damaged items?"
}
```

**Response**:
```json
{
  "answer": "Based on the retrieved context: Grocery and perishable items may be reported for a return within 24 hours of delivery if damaged, spoiled, or incorrect; non-perishable packaged items may be returned within 7 days of delivery...",
  "sources": [
    "doc_02"
  ],
  "confidence": 1.0
}
```

### Example 2: General Question (Direct Answer)
**Request (`POST /ask`)**:
```json
{
  "query": "Who is the Prime Minister of India?"
}
```

**Response**:
```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```