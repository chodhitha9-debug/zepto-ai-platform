import os
import glob
from typing import TypedDict, List, Optional
MOCK_LLM = os.getenv("MOCK_LLM", "1")
from pydantic import BaseModel, Field
from fastapi import FastAPI
import uvicorn
import chromadb
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END

# --- Pydantic Schema ---
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
    confidence: float

# --- ChromaDB Setup ---
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
        return {
            "model_name": "all-MiniLM-L6-v2"
        }

    @staticmethod
    def build_from_config(config):
        return LocalEmbeddingFunction()
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])

        start = end - overlap

    return chunks

chroma_client = chromadb.PersistentClient(path=DB_DIR)
collection = chroma_client.get_or_create_collection(
    name="zepto_policies",
    embedding_function=LocalEmbeddingFunction()
)

def init_vector_store():
    if collection.count() == 0 and os.path.exists(DOCS_DIR):

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

                metadatas.append({
                    "source": doc_id,
                    "chunk_id": chunk_id
                })

        if documents:
            collection.add(
                documents=documents,
                ids=ids,
                metadatas=metadatas
            )
init_vector_store()

# --- LangGraph Setup ---
class GraphState(TypedDict):
    query: str
    intent: Optional[str]
    retrieved_docs: Optional[List[str]]
    retrieved_ids: Optional[List[str]]
    final_response: Optional[QueryResponse]

def classify_intent(state: GraphState) -> GraphState:
    query = state["query"].lower()
    keywords = ["delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours"]
    if any(kw in query for kw in keywords):
        intent = "policy_question"
    else:
        intent = "general_question"
    return {**state, "intent": intent}
STRUCTURED_PROMPT = """
Role:
You are a Zepto customer support assistant.

Context:
Answer only using the policy information provided below.

Task:
Answer the user's question clearly and directly.

Format:
Return a short, helpful answer.

Length:
Keep the answer concise.

Constraint:
Do not answer using information that is not present in the provided context.

Example:
User: What is the delivery fee for orders under INR 500?
Context: Orders under INR 500 have a delivery fee of INR 49.
Answer: The delivery fee is INR 49 for orders under INR 500.

User question:
{query}

Provided context:
{context}
"""

def retrieve_and_answer(state: GraphState) -> GraphState:
    query = state["query"]
    results = collection.query(query_texts=[query], n_results=3)
    retrieved_docs = results["documents"][0] if results["documents"] else []
    retrieved_ids = results["ids"][0] if results["ids"] else []
    
    context = "\n".join(retrieved_docs)

    prompt = STRUCTURED_PROMPT.format(
        query=query,
        context=context
    )

    if MOCK_LLM == "1":
        if retrieved_docs:
        # Deterministic mock generation using the structured prompt
            answer = (
                "Based on the provided policy context: "
                + retrieved_docs[0]
            )
        else:
            answer = "No relevant policy information was found."
    else:
        answer = (
            f"LLM mode is enabled. Prompt prepared for the real LLM:\n{prompt}"
        )
    
    response = QueryResponse(
        answer=answer,
        sources=retrieved_ids,
        confidence=0.9 if retrieved_docs else 0.0
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

    response = QueryResponse(
        answer=answer,
        sources=[],
        confidence=1.0
    )

    return {**state, "final_response": response}

def route_intent(state: GraphState) -> str:
    return state["intent"]

builder = StateGraph(GraphState)
builder.add_node("classify_intent", classify_intent)
builder.add_node("retrieve_and_answer", retrieve_and_answer)
builder.add_node("direct_answer", direct_answer)

builder.set_entry_point("classify_intent")
builder.add_conditional_edges("classify_intent", route_intent, {
    "policy_question": "retrieve_and_answer",
    "general_question": "direct_answer"
})
builder.add_edge("retrieve_and_answer", END)
builder.add_edge("direct_answer", END)

graph = builder.compile()

# --- FastAPI App ---
app = FastAPI(title="Zepto Support Assistant")

@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    initial_state = {
        "query": request.query,
        "intent": None,
        "retrieved_docs": None,
        "retrieved_ids": None,
        "final_response": None
    }
    result = graph.invoke(initial_state)
    return result["final_response"]

if __name__ == "__main__":
    uvicorn.run("assistant:app", host="0.0.0.0", port=7860, reload=True)