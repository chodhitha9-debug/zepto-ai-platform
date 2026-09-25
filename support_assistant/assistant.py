import os
import glob
from typing import TypedDict, List, Optional

import chromadb
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field, ValidationError
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END

MOCK_LLM = os.getenv("MOCK_LLM", "1")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)

DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

class LocalEmbeddingFunction:
    def __call__(self, input: List[str]) -> List[List[float]]:
        return embedding_model.encode(input).tolist()

    def embed_query(self, input: List[str]) -> List[List[float]]:
        return embedding_model.encode(input).tolist()

    @staticmethod
    def name() -> str:
        return "local_sentence_transformer"

    def get_config(self):
        return {"model_name": "all-MiniLM-L6-v2"}

    @staticmethod
    def build_from_config(config):
        return LocalEmbeddingFunction()

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])

        if end >= len(text):
            break

        start = end - overlap

    return chunks

chroma_client = chromadb.PersistentClient(path=DB_DIR)

collection = chroma_client.get_or_create_collection(
    name="zepto_policies",
    embedding_function=LocalEmbeddingFunction()
)

def init_vector_store():
    if collection.count() > 0 or not os.path.exists(DOCS_DIR):
        return

    doc_files = sorted(
        glob.glob(os.path.join(DOCS_DIR, "doc_*.txt"))
    )

    documents = []
    ids = []
    metadatas = []

    for file_path in doc_files:
        doc_id = os.path.basename(file_path).replace(".txt", "")

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        chunks = chunk_text(text)

        for chunk_number, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{chunk_number}"

            documents.append(chunk)
            ids.append(chunk_id)
            metadatas.append(
                {
                    "source": doc_id,
                    "chunk_id": chunk_id
                }
            )

    if documents:
        collection.add(
            documents=documents,
            ids=ids,
            metadatas=metadatas
        )

init_vector_store()

class GraphState(TypedDict):
    query: str
    intent: Optional[str]
    retrieved_docs: Optional[List[str]]
    retrieved_ids: Optional[List[str]]
    final_response: Optional[QueryResponse]

def classify_intent(state: GraphState) -> GraphState:
    query = state["query"].lower()

    keywords = [
        "delivery",
        "return",
        "refund",
        "membership",
        "tracking",
        "track",
        "cancel",
        "cancellation",
        "gift card",
        "giftcard",
        "support hours",
        "support",
        "phone support",
        "email support",
        "priority delivery",
        "standard delivery",
        "damaged",
        "spoiled",
        "missing",
        "replacement"
    ]

    if any(keyword in query for keyword in keywords):
        intent = "policy_question"
    else:
        intent = "general_question"

    return {
        **state,
        "intent": intent
    }

STRUCTURED_PROMPT = """
Role:
You are a Zepto customer support assistant.

Context:
Answer only using the policy information provided below.

Task:
Answer the user's question clearly and directly using the provided policy context.

Format:
Return a short, helpful answer and do not invent policy details.

Length:
Keep the answer concise and relevant to the user's question.

Negative Constraint:
Do not use information that is not present in the provided context. Do not invent prices, time periods, eligibility rules, exceptions, or procedures.

Few-shot Example:
User: What is the delivery fee for orders under INR 500?
Context: Orders under INR 500 incur a delivery fee of INR 49.
Answer: The delivery fee is INR 49 for orders under INR 500.

User question:
{query}

Provided context:
{context}
"""

def build_mock_policy_answer(query: str, retrieved_docs: List[str]) -> str:
    if not retrieved_docs:
        return "No relevant policy information was found."

    return "Based on the provided policy context: " + retrieved_docs[0]

def validate_response_with_retry(
    answer: str,
    sources: List[str],
    confidence: float,
    retry_answer: Optional[str] = None
) -> QueryResponse:
    try:
        return QueryResponse(
            answer=answer,
            sources=sources,
            confidence=confidence
        )
    except ValidationError:
        if retry_answer is None:
            retry_answer = "I could not generate a valid response."

        return QueryResponse(
            answer=retry_answer,
            sources=sources,
            confidence=0.0
        )

def retrieve_and_answer(state: GraphState) -> GraphState:
    query = state["query"]

    results = collection.query(
        query_texts=[query],
        n_results=3
    )

    retrieved_docs = (
        results["documents"][0]
        if results.get("documents")
        else []
    )

    retrieved_ids = (
        results["ids"][0]
        if results.get("ids")
        else []
    )

    context = "\n".join(retrieved_docs)

    prompt = STRUCTURED_PROMPT.format(
        query=query,
        context=context
    )

    if MOCK_LLM == "1":
        answer = build_mock_policy_answer(
            query,
            retrieved_docs
        )

        response = validate_response_with_retry(
            answer=answer,
            sources=retrieved_ids,
            confidence=0.9 if retrieved_docs else 0.0,
            retry_answer="No relevant policy information was found."
        )
    else:
        generated_answer = (
            f"LLM mode is enabled. Prompt prepared for the real LLM:\n{prompt}"
        )

        response = validate_response_with_retry(
            answer=generated_answer,
            sources=retrieved_ids,
            confidence=0.9 if retrieved_docs else 0.0,
            retry_answer="I could not generate a valid policy response."
        )

    return {
        **state,
        "retrieved_docs": retrieved_docs,
        "retrieved_ids": retrieved_ids,
        "final_response": response
    }

def direct_answer(state: GraphState) -> GraphState:
    if MOCK_LLM == "1":
        answer = "I can only answer questions about Zepto policies right now."
    else:
        answer = (
            "LLM mode is enabled. A real LLM can be connected here "
            "for general questions."
        )

    response = validate_response_with_retry(
        answer=answer,
        sources=[],
        confidence=1.0,
        retry_answer="I can only answer questions about Zepto policies right now."
    )

    return {
        **state,
        "final_response": response
    }

def route_intent(state: GraphState) -> str:
    return state["intent"]

builder = StateGraph(GraphState)

builder.add_node(
    "classify_intent",
    classify_intent
)

builder.add_node(
    "retrieve_and_answer",
    retrieve_and_answer
)

builder.add_node(
    "direct_answer",
    direct_answer
)

builder.set_entry_point("classify_intent")

builder.add_conditional_edges(
    "classify_intent",
    route_intent,
    {
        "policy_question": "retrieve_and_answer",
        "general_question": "direct_answer"
    }
)

builder.add_edge(
    "retrieve_and_answer",
    END
)

builder.add_edge(
    "direct_answer",
    END
)

graph = builder.compile()

app = FastAPI(
    title="Zepto Support Assistant"
)

@app.post(
    "/ask",
    response_model=QueryResponse
)
def ask_question(request: QueryRequest):
    initial_state: GraphState = {
        "query": request.query,
        "intent": None,
        "retrieved_docs": None,
        "retrieved_ids": None,
        "final_response": None
    }

    result = graph.invoke(initial_state)

    return result["final_response"]

if __name__ == "__main__":
    uvicorn.run(
        "assistant:app",
        host="0.0.0.0",
        port=7860,
        reload=True
    )