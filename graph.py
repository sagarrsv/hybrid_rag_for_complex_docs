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
import time
from context_builder import ContextBuilder
from config import GENERATOR_CONTEXT_PAGES


class RAGState(TypedDict, total=False):
    # User input
    mode: str
    query: str
    k: int

    # Retrieval stage
    retrieved_docs: List[Dict[str, Any]]
    retrieval_latency: float

    # Context construction stage
    assembled_context: Dict[str, Any]
    context_latency: float

    # Generation stage
    answer: str
    answerable: bool
    citations: List[Dict[str, Any]]
    usage: Dict[str, Any]
    generation_latency: float



def build_rag_graph(
    retriever: MultimodalRetriever,
    generator: MultimodalGenerator,
    context_builder: ContextBuilder,
):
    """
    Simple RAG pipeline:

    retriever -> context_builder -> generator

    Retrieval scoring and fusion are unchanged.
    """

    def retrieve_node(state: RAGState):
        mode = state.get(
            "mode",
            "hybrid_v2_d3",
        )

        query = state["query"]

        k = state.get(
            "k",
            GENERATOR_CONTEXT_PAGES,
        )

        hits, latency = retriever.search(
            mode,
            query,
            k=k,
        )

        return {
            "retrieved_docs": hits,
            "retrieval_latency": latency,
        }

    def context_node(state: RAGState):
        start = time.perf_counter()

        context = context_builder.build(
            retrieved_docs=state["retrieved_docs"],
        )

        return {
            "assembled_context": context,
            "context_latency":
                time.perf_counter() - start,
        }

    def generate_node(state: RAGState):
        result = generator.generate(
            query=state["query"],
            context=state["assembled_context"],
        )

        return {
            "answer": result["answer"],
            "answerable": result["answerable"],
            "citations": result["citations"],
            "usage": result["usage"],
            "generation_latency":
                result["generation_latency"],
        }

    workflow = StateGraph(RAGState)

    workflow.add_node(
        "retriever",
        retrieve_node,
    )

    workflow.add_node(
        "context_builder",
        context_node,
    )

    workflow.add_node(
        "generator",
        generate_node,
    )

    workflow.set_entry_point("retriever")

    workflow.add_edge(
        "retriever",
        "context_builder",
    )

    workflow.add_edge(
        "context_builder",
        "generator",
    )

    workflow.add_edge("generator", END)

    return workflow.compile()
