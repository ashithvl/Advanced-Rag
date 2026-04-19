"""End-to-end RAG pipeline orchestrated with LangGraph.

Nodes:
    expand    -> rewrite query into N variants (HyDE / multi-query)
    retrieve  -> hybrid dense + sparse fusion + rerank
    gate      -> confidence gate; route to answer or fallback
    answer    -> grounded generation with citations
    verify    -> chain-of-verification; mark unsupported claims

The graph is intentionally small and explicit so it is easy to extend with
graph-RAG (multi-hop on Neo4j) or vision retrieval branches.
"""

from __future__ import annotations

import json
import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from advanced_rag.config import get_settings
from advanced_rag.generation import GROUNDED_ANSWER_PROMPT, get_chat_llm
from advanced_rag.generation.prompts import VERIFICATION_PROMPT


class RagState(TypedDict, total=False):
    question: str
    expanded_queries: Annotated[list[str], operator.add]
    contexts: list[dict[str, Any]]
    answer: str
    faithful: bool
    unsupported: list[str]
    confidence: float


def _expand(state: RagState) -> RagState:
    llm = get_chat_llm(temperature=0.2)
    msg = HumanMessage(
        content=(
            "Rewrite the following technical question as 3 alternative search queries "
            "(one per line, no numbering):\n\n" + state["question"]
        )
    )
    raw = llm.invoke([msg]).content
    variants = [line.strip() for line in str(raw).splitlines() if line.strip()]
    return {"expanded_queries": [state["question"], *variants]}


def _retrieve(state: RagState) -> RagState:
    """Hybrid retrieval + rerank.

    Imports are lazy so the graph can be imported without a live Weaviate
    connection (e.g., during tests or when only `expand`/`answer` are used).
    """
    from advanced_rag.retrieval import build_hybrid_retriever, get_reranker

    retriever = build_hybrid_retriever()
    reranker = get_reranker()

    seen: dict[str, Any] = {}
    for q in state.get("expanded_queries") or [state["question"]]:
        for node in retriever.retrieve(q):
            seen.setdefault(node.node.node_id, node)

    nodes = list(seen.values())
    reranked = reranker.postprocess_nodes(nodes, query_str=state["question"])

    contexts = [
        {
            "id": n.node.node_id,
            "text": n.node.get_content(),
            "metadata": n.node.metadata,
            "score": float(n.score or 0.0),
        }
        for n in reranked
    ]
    top_score = max((c["score"] for c in contexts), default=0.0)
    return {"contexts": contexts, "confidence": top_score}


def _gate(state: RagState) -> str:
    threshold = get_settings().confidence_threshold
    return "answer" if state.get("confidence", 0.0) >= threshold else "fallback"


def _answer(state: RagState) -> RagState:
    llm = get_chat_llm()
    parts = []
    for c in state.get("contexts", []):
        meta = c["metadata"]
        header = f"[{meta.get('file_name', '?')} p.{meta.get('page_label', '?')}]"
        parts.append(f"{header}\n{c['text']}")
    context_block = "\n\n---\n\n".join(parts)
    chain = GROUNDED_ANSWER_PROMPT | llm
    out = chain.invoke({"question": state["question"], "context": context_block})
    return {"answer": str(out.content)}


def _fallback(state: RagState) -> RagState:
    return {
        "answer": "Not specified in the manual.",
        "faithful": True,
        "unsupported": [],
    }


def _verify(state: RagState) -> RagState:
    llm = get_chat_llm(temperature=0.0)
    context_block = "\n\n---\n\n".join(c["text"] for c in state.get("contexts", []))
    chain = VERIFICATION_PROMPT | llm
    raw = chain.invoke({"context": context_block, "answer": state["answer"]}).content
    try:
        parsed = json.loads(str(raw))
        return {
            "faithful": bool(parsed.get("faithful", False)),
            "unsupported": list(parsed.get("unsupported", [])),
        }
    except json.JSONDecodeError:
        return {"faithful": False, "unsupported": ["<verifier returned non-JSON>"]}


def build_graph() -> Any:
    g = StateGraph(RagState)
    g.add_node("expand", _expand)
    g.add_node("retrieve", _retrieve)
    g.add_node("answer", _answer)
    g.add_node("fallback", _fallback)
    g.add_node("verify", _verify)

    g.set_entry_point("expand")
    g.add_edge("expand", "retrieve")
    g.add_conditional_edges("retrieve", _gate, {"answer": "answer", "fallback": "fallback"})
    g.add_edge("answer", "verify")
    g.add_edge("verify", END)
    g.add_edge("fallback", END)
    return g.compile()


def run(question: str) -> RagState:
    """Convenience entry point for FastAPI / Streamlit / scripts."""
    return build_graph().invoke({"question": question})
