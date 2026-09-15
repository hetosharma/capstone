"""LangGraph intent routing, deterministic mock generation, and optional live generation."""

from __future__ import annotations

import json
import os
from typing import TypedDict

import chromadb
from pydantic import BaseModel, Field, ValidationError
from sentence_transformers import SentenceTransformer
from langgraph.graph import END, StateGraph

from .prompts import build_prompt

ROOT = os.path.dirname(__file__)
KEYWORDS = ("delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours")


class AnswerResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


class AssistantState(TypedDict, total=False):
    query: str
    intent: str
    chunks: list[dict[str, str]]
    response: AnswerResponse


def mock_mode() -> bool:
    return os.getenv("MOCK_LLM", "1") != "0"


class Retriever:
    def __init__(self) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        client = chromadb.PersistentClient(path=os.path.join(ROOT, "chroma_db"))
        self.collection = client.get_or_create_collection("zepto_support_policies", metadata={"hnsw:space": "cosine"})

    def search(self, query: str, k: int = 3) -> list[dict[str, str]]:
        result = self.collection.query(query_embeddings=[self.model.encode(query, normalize_embeddings=True).tolist()], n_results=k)
        docs = result.get("documents", [[]])[0]
        ids = result.get("ids", [[]])[0]
        return [{"id": str(identifier), "text": str(document)} for identifier, document in zip(ids, docs)]


def classify_intent(state: AssistantState) -> AssistantState:
    query = state["query"].lower()
    return {**state, "intent": "policy_question" if any(keyword in query for keyword in KEYWORDS) else "general_question"}


def live_answer(question: str, chunks: list[dict[str, str]]) -> AnswerResponse:
    """Optional Groq-compatible path; mock mode never reaches this function."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return AnswerResponse(answer="Live LLM mode requires GROQ_API_KEY.", sources=[], confidence=0.0)
    from groq import Groq
    client = Groq(api_key=api_key)
    prompt = build_prompt(question, chunks)
    last_error: Exception | None = None
    for attempt in range(3):
        instruction = prompt if attempt == 0 else prompt + "\nCorrective instruction: return only valid JSON matching the required schema."
        try:
            raw = client.chat.completions.create(
                model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
                messages=[{"role": "user", "content": instruction}],
                temperature=0,
                response_format={"type": "json_object"},
            ).choices[0].message.content
            return AnswerResponse.model_validate(json.loads(raw))
        except (ValidationError, json.JSONDecodeError, TypeError, AttributeError) as exc:
            last_error = exc
    return AnswerResponse(answer=f"LLM response validation failed after 3 attempts: {last_error}", sources=[], confidence=0.0)


def retrieve_and_answer(state: AssistantState) -> AssistantState:
    chunks = Retriever().search(state["query"])
    if mock_mode():
        top = chunks[0] if chunks else {"id": "none", "text": "No policy context was retrieved."}
        response = AnswerResponse(answer=f"Based on the retrieved context: {top['text'][:200]}", sources=[c["id"] for c in chunks], confidence=1.0)
    else:
        response = live_answer(state["query"], chunks)
    return {**state, "chunks": chunks, "response": response}


def direct_answer(state: AssistantState) -> AssistantState:
    if mock_mode():
        response = AnswerResponse(answer="I can only answer questions about Zepto policies right now.", sources=[], confidence=1.0)
    else:
        response = live_answer(state["query"], [])
    return {**state, "response": response}


def route(state: AssistantState) -> str:
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"


def build_graph():
    graph = StateGraph(AssistantState)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)
    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges("classify_intent", route, {"retrieve_and_answer": "retrieve_and_answer", "direct_answer": "direct_answer"})
    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)
    return graph.compile()


assistant_graph = build_graph()


