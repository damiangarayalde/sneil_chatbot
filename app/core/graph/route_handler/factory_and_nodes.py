from __future__ import annotations

import time
from functools import lru_cache

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from app.core.graph.state import (
    ChatState,
    clear_lock,
    reset_solve_state,
    get_history_and_last_msg,
    get_last_msg,
)
from app.core.tools.rag import get_retriever
from app.core.tools.catalog_tool import catalog_lookup

from .chain import get_route_chain_safe_invoke, get_tool_router_llm
from app.core.logging_config import get_logger

_logger = get_logger("sneil.handler")


@lru_cache(maxsize=64)
def _get_route_retriever(route_id: str, k: int = 3):
    return get_retriever(route_id, k=k)


def route_node_name(route: str) -> str:
    """Canonical node name for a route handler."""
    return f"handle__{route}"


def _invoke_tool_router(route_id: str, last_msg: str) -> tuple[str, list[dict]]:
    """Call the tool-router LLM; execute its tool calls; return (context_str, raw_docs).

    The LLM decides whether to call catalog_lookup, rag_retrieval, or neither.
    On any failure, returns ("", []) so generate() can still run without context.
    """
    from app.core.tools.catalog_tool_llm import create_catalog_lookup_tool
    from app.core.tools.rag_tool_llm import create_rag_retrieval_tool

    try:
        catalog_tool = create_catalog_lookup_tool()
        rag_tool = create_rag_retrieval_tool(_get_route_retriever, route_id)
        tool_router = get_tool_router_llm([catalog_tool, rag_tool])
        response = tool_router.invoke([HumanMessage(content=last_msg)])
    except Exception:
        _logger.warning(
            "tool router failed, proceeding without tool context",
            extra={"route": route_id},
        )
        return "", []

    result_parts: list[str] = []
    raw_docs: list[dict] = []

    for tc in (getattr(response, "tool_calls", None) or []):
        name = tc.get("name", "")
        args = tc.get("args", {})

        if name == "catalog_lookup":
            out = catalog_lookup(
                args.get("query", last_msg),
                product_family=route_id,
                k=args.get("k", 3),
            )
            _logger.info(
                "catalog lookup via tool router",
                extra={"route": route_id, "matches": out["count"]},
            )
            result_parts.append("\n\nCATALOG_LOOKUP:\n" + str(out))

        elif name == "rag_retrieval":
            retriever = _get_route_retriever(route_id, k=args.get("k", 3))
            docs = retriever.invoke(args.get("query", last_msg))
            if docs:
                raw_docs = [
                    {"page_content": d.page_content, "metadata": d.metadata}
                    for d in docs
                ]
                result_parts.append(
                    "\n\nRAG_DOCS:\n" + "\n\n".join(d.page_content for d in docs)
                )

    return "".join(result_parts), raw_docs


def make_route_subgraph(route_id: str):
    """Construct a StateGraph subgraph for a given locked route.

    Flow: START → generate → END

    The LLM decides which tools (catalog_lookup, rag_retrieval, or neither)
    to invoke via bind_tools before generating the final answer.
    """

    def generate(state: ChatState) -> ChatState:
        """Generate an answer using the LLM and any tool-retrieved context."""
        t0 = time.perf_counter()

        history, last_msg = get_history_and_last_msg(state.get("messages") or [])

        # Tool-router LLM decides whether to call catalog_lookup, rag_retrieval, or neither
        context_text, raw_docs = _invoke_tool_router(route_id, last_msg)

        _logger.debug(
            "context assembled",
            extra={"node": f"generate__{route_id}", "route": route_id},
        )

        # Invoke handler chain with retry, timeout, and fallback
        response = get_route_chain_safe_invoke(
            route_id,
            {
                "history": history,
                "user_text": last_msg,
                "context": context_text,
                "meta": "",
            },
        )

        # Topic switch => clear lock so classifier can re-route next turn
        if response.is_topic_switch:
            return {
                **clear_lock(),
                **reset_solve_state(),
            }

        dt_ms = round((time.perf_counter() - t0) * 1000, 1)
        _logger.info(
            "LLM generate complete",
            extra={"node": f"generate__{route_id}", "route": route_id, "duration_ms": dt_ms},
        )

        # Increment solve_attempts only if LLM detects previous solution was ineffective
        current_attempts = int(state.get("solve_attempts") or 0)
        solve_attempts_so_far = (
            current_attempts + 1
            if getattr(response, "increment_solve_attempts", False)
            else current_attempts
        )

        answer_text = (response.answer or "").strip()
        if not answer_text:
            answer_text = "Entiendo. ¿Podés darme un poco más de detalle para ayudarte mejor?"

        return {
            "messages": [AIMessage(content=answer_text)],
            "solve_attempts": solve_attempts_so_far,
            "retrieved": raw_docs,  # preserved for observability and test assertions
        }

    g = StateGraph(ChatState)
    g.add_node("generate", generate)
    g.add_edge(START, "generate")
    g.add_edge("generate", END)
    return g.compile()
