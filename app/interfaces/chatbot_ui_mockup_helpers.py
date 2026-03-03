"""
Shared helpers for the WhatsApp-style dev UI (chatbot_ui_mockup.html).

These utilities are used by:
- app.interfaces.api_test_route_subgraph
- app.interfaces.dev_api

Goal: keep the dev APIs tiny and consistent while avoiding copy/paste drift.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from app.core.graph.flow_logging import wrap_node
from app.core.graph.route_handler.factory_and_nodes import make_route_subgraph
from app.core.graph.state import ChatState


def validate_route(route_id: str, routes: set[str]) -> str:
    """Validate a route_id against the known routes and return it (or raise)."""
    if route_id not in routes:
        raise RuntimeError(
            f"Invalid TEST_ROUTE={route_id!r}. Valid routes: {sorted(routes)}"
        )
    return route_id


def build_route_only_graph(route_id: str, checkpointer: Optional[Any] = None):
    """Compile a tiny graph that runs a single route subgraph once per turn."""
    subgraph = make_route_subgraph(route_id)

    g = StateGraph(ChatState)
    node_name = f"handle__{route_id}"
    g.add_node(node_name, wrap_node(node_name, subgraph))

    g.add_edge(START, node_name)
    g.add_edge(node_name, END)

    if checkpointer is not None:
        return g.compile(checkpointer=checkpointer)
    return g.compile()


def html_path_for(py_file: str, html_filename: str = "chatbot_ui_mockup.html") -> Path:
    """Return the expected UI HTML path next to the given python file."""
    return Path(py_file).resolve().parent / html_filename


def render_page(
    py_file: str,
    test_route: str,
    html_filename: str = "chatbot_ui_mockup.html",
) -> str:
    """
    Load and return the HTML UI.

    Uses simple placeholder replacement (no f-strings, no brace escaping issues):
      {{TEST_ROUTE}} -> test_route
    """
    html_path = html_path_for(py_file, html_filename)
    if not html_path.exists():
        return (
            "<h2>Missing UI file</h2>"
            f"<p>Expected to find: <code>{html_path}</code></p>"
            f"<p>Copy <code>{html_filename}</code> next to <code>{Path(py_file).name}</code>.</p>"
        )

    html = html_path.read_text(encoding="utf-8")
    return html.replace("{{TEST_ROUTE}}", test_route)


def extract_assistant_text(output: Dict[str, Any]) -> str:
    """
    Collect all AI messages produced after the latest HumanMessage in the returned
    state (helps when the graph emits multiple AI messages per turn).
    """
    msgs = output.get("messages") or []

    last_human_idx = next(
        (
            i
            for i in range(len(msgs) - 1, -1, -1)
            if isinstance(msgs[i], HumanMessage)
        ),
        None,
    )
    start = (last_human_idx + 1) if last_human_idx is not None else 0
    ai_after = [m for m in msgs[start:] if isinstance(m, AIMessage)]

    parts = []
    for m in ai_after:
        content = (m.content or "").strip()
        if content:
            parts.append(content)

    return "\n\n".join(parts).strip()


def make_config(thread_id: str) -> dict:
    """LangGraph config wrapper for the checkpointer thread id."""
    return {"configurable": {"thread_id": thread_id}}


def reset_thread_state(
    graph: Any,
    thread_id: str,
    locked_route: str,
    *,
    attempts: int = 0,
) -> None:
    """
    Clears persisted state for this thread (messages + attempts) so the UI's
    /reset button is a true "fresh chat" for the same thread_id.
    """
    config = make_config(thread_id)
    clean_state = {
        "messages": [],
        "locked_route": locked_route,
        "attempts": attempts,
    }

    if hasattr(graph, "update_state"):
        try:
            # type: ignore[attr-defined]
            graph.update_state(config, clean_state)
            return
        except TypeError:
            # Some langgraph versions require as_node=START
            # type: ignore[attr-defined]
            graph.update_state(config, clean_state, as_node=START)
            return
