from __future__ import annotations

import time
from functools import lru_cache
from typing import Literal
from urllib import response

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

from app.core.graph.msg_heuristics_no_llm import should_retrieve
from app.core.graph.state import (
    ChatState,
    clear_lock,
    reset_solve_state,
    get_history_and_last_msg,
    get_last_msg,
)
from app.core.tools.rag import get_retriever
from app.core.tools.catalog_tool import catalog_lookup

from .chain import get_route_chain


@lru_cache(maxsize=64)
def _get_route_retriever(route_id: str, k: int = 3):
    return get_retriever(route_id, k=k)


def route_node_name(route: str) -> str:
    """Canonical node name for a route handler."""
    return f"handle__{route}"


def make_route_subgraph(route_id: str):
    """Construct a StateGraph subgraph for a given locked route.

    Intended behavior:
    1) START chooses retrieve vs generate based on heuristics.
    2) retrieve (optional) -> generate -> END
    """

    def route_from_start(state: ChatState) -> Literal["retrieve", "generate"]:
        last_msg = get_last_msg(state.get("messages") or [])
        return "retrieve" if should_retrieve(last_msg) else "generate"

    def retrieve(state: ChatState) -> ChatState:
        """Retrieve context documents for the route based on the user query."""
        last_msg = get_last_msg(state.get("messages") or [])
        retriever = _get_route_retriever(route_id, k=3)
        retrieved_docs = retriever.invoke(last_msg)
        return {"retrieved": [{"page_content": d.page_content, "metadata": d.metadata} for d in retrieved_docs]}

    def generate(state: ChatState) -> ChatState:
        """Generate an answer using the LLM and retrieved context."""
        t0 = time.perf_counter()
        chain = get_route_chain(route_id)

        history, last_msg = get_history_and_last_msg(
            state.get("messages") or [])

        # Prepare context string from retrieved docs
        docs = state.get("retrieved") or []
        context_text = "\n\n".join((d.get("page_content") or "") for d in docs)

        # Check if user query involves catalog-related keywords
        # in the future, instead of providing a list of escape words like here a better approach would be to enable access to the catalog as a skill for the llm to decide wheter to query it based on the user msg.
        if any(x in last_msg.lower() for x in ["precio", "valor", "sale", "vale", "cuesta", "link", "comprar", "sku"]):
            tool_out = catalog_lookup(
                last_msg, product_family=route_id, k=3)
            print(f"Found {tool_out['count']} matches (showing top {3}):")
            for i, match in enumerate(tool_out['matches'], 1):
                price = match.get('price', 'N/A')
                print(
                    f"  {i}. [{match.get('id')}] {match.get('title')} - ${price} {tool_out['currency']}")
            context_text += "\n\nCATALOG_LOOKUP:\n" + str(tool_out)

        print(context_text)

        response = chain.invoke(
            {
                "history": history,
                "user_text": last_msg,
                "context": context_text,
                "meta": "",
            }
        )

        # Topic switch => clear lock so classifier can re-route next
        if response.is_topic_switch:
            return {
                **clear_lock(),
                **reset_solve_state(),
            }

        dt_ms = (time.perf_counter() - t0) * 1000
        print(f"[{route_id}] LLM elapsed: {dt_ms:.1f} ms")

        # Increment solve_attempts only if LLM detects previous solution was ineffective
        current_attempts = int(state.get("solve_attempts") or 0)
        solve_attempts_so_far = current_attempts + \
            1 if getattr(response, 'increment_solve_attempts',
                         False) else current_attempts

        answer_text = (response.answer or "").strip()
        if not answer_text:
            answer_text = "Entiendo. ¿Podés darme un poco más de detalle para ayudarte mejor?"

        return {
            "messages": [AIMessage(content=answer_text)],
            "solve_attempts": solve_attempts_so_far,
        }

    g = StateGraph(ChatState)

    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)

    g.add_conditional_edges(
        START,
        route_from_start,
        {"retrieve": "retrieve", "generate": "generate"},
    )
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)

    return g.compile()
