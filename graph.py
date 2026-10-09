"""
graph.py
====================================================================
Single-mode baseline LangGraph workflow.
Decoupled retrieval and generation nodes with execution latency tracking.
====================================================================
"""

from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from retriever import MultimodalRetriever
from generator import MultimodalGenerator
from config import MAX_K


class RAGState(TypedDict):
    mode: str
    query: str
    k: int
    retrieved_docs: List[Dict[str, Any]]
    retrieval_latency: float
    answer: str


def build_rag_graph(retriever: MultimodalRetriever, generator: MultimodalGenerator):
    def retrieve_node(state: RAGState) -> Dict[str, Any]:
        mode = state.get("mode", "hybrid_routed")
        query = state["query"]
        k = state.get("k", MAX_K)

        hits, latency = retriever.search(mode, query, k=k)
        return {
            "retrieved_docs": hits,
            "retrieval_latency": latency
        }

    def generate_node(state: RAGState) -> Dict[str, Any]:
        ans = generator.generate(state["query"], state["retrieved_docs"])
        return {"answer": ans}

    workflow = StateGraph(RAGState)
    workflow.add_node("retriever", retrieve_node)
    workflow.add_node("generator", generate_node)

    workflow.set_entry_point("retriever")
    workflow.add_edge("retriever", "generator")
    workflow.add_edge("generator", END)

    return workflow.compile()